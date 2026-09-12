// Package controller holds the Kubernetes reconciler for InferenceCluster.
//
// The controller owns *convergence*. It owns no rules: placement comes from
// internal/scheduler and host-phase legality from internal/fleet, both of which
// are pure and unit-tested without an API server. That split is deliberate —
// the interesting logic should not require a cluster to test.
//
// Two properties matter more than anything else here:
//
//   - It converges without churning. Existing placements are honoured if they
//     are still valid; only the delta is placed. A controller that re-runs the
//     packer from scratch every pass will happily migrate live replicas because
//     the packer found a marginally tighter arrangement.
//   - Status describes the generation it was computed from. ObservedGeneration
//     trailing metadata.generation is how a caller distinguishes "not converged
//     yet" from "converged, and this is the answer".
package controller

import (
	"context"
	"fmt"
	"strconv"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	fleetv1 "github.com/rajj28/gpu-fleet-operator/api/v1alpha1"
	"github.com/rajj28/gpu-fleet-operator/internal/scheduler"
)

const (
	// LabelGPUType marks which hardware pool a node belongs to.
	LabelGPUType = "fleet.gpu.io/gpu-type"
	// LabelDomain marks the node's interconnect domain (NVLink island, rack,
	// fabric partition). Nodes without it land in domain "" together, which is
	// correct for a flat fleet and wrong quietly for a segmented one — so the
	// controller logs when it sees a mix.
	LabelDomain = "fleet.gpu.io/domain"
	// ResourceGPU is the extended resource the device plugin advertises.
	ResourceGPU corev1.ResourceName = "nvidia.com/gpu"
	// Finalizer keeps the object around long enough to release capacity.
	Finalizer = "fleet.gpu.io/drain"

	// requeueDegraded is how long to wait before re-checking a cluster that is
	// short of capacity. Capacity frees up when other clusters shrink, and
	// there is no watch event for "someone else's replica went away" that this
	// controller does not already get — but a bounded retry is cheap insurance.
	requeueDegraded = 30 * time.Second
)

// InferenceClusterReconciler reconciles InferenceCluster objects.
type InferenceClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=fleet.gpu.io,resources=inferenceclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=fleet.gpu.io,resources=inferenceclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=fleet.gpu.io,resources=inferenceclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch

// Reconcile drives one InferenceCluster towards its spec.
func (r *InferenceClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var ic fleetv1.InferenceCluster
	if err := r.Get(ctx, req.NamespacedName, &ic); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil // deleted and finalized; nothing owed
		}
		return ctrl.Result{}, err
	}

	// --- deletion: drain before letting go ---
	if !ic.DeletionTimestamp.IsZero() {
		return r.drain(ctx, &ic)
	}
	if !controllerutil.ContainsFinalizer(&ic, Finalizer) {
		controllerutil.AddFinalizer(&ic, Finalizer)
		if err := r.Update(ctx, &ic); err != nil {
			return ctrl.Result{}, err
		}
		// The update bumps resourceVersion; requeue and work from fresh state
		// rather than reasoning about a copy we have already mutated.
		return ctrl.Result{Requeue: true}, nil
	}

	// --- build the fleet view from nodes ---
	fleet, err := r.fleetFromNodes(ctx)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("building fleet view: %w", err)
	}

	// --- occupy it with what every *other* cluster already holds ---
	var all fleetv1.InferenceClusterList
	if err := r.List(ctx, &all); err != nil {
		return ctrl.Result{}, fmt.Errorf("listing inference clusters: %w", err)
	}
	gpuType := scheduler.GPUType(ic.Spec.GPUType)
	for i := range all.Items {
		other := &all.Items[i]
		// Identify by namespace/name, not UID. UID is assigned by the API
		// server and is empty for objects that have not been through it — so a
		// UID comparison silently matches *every* cluster against every other,
		// which made the fleet view look empty and double-booked every node.
		if other.Namespace == ic.Namespace && other.Name == ic.Name {
			continue
		}
		occupy(fleet, other)
	}

	// --- honour the placements we already have, then place only the delta ---
	want := int(ic.Spec.Replicas)
	size := int(ic.Spec.GPUsPerReplica)
	kept := make([]fleetv1.Placement, 0, want)

	for _, p := range ic.Status.Placements {
		if len(kept) >= want {
			break // scaled down: the rest are released by simply not keeping them
		}
		w := scheduler.Workload{Name: replicaName(&ic, p.Replica), Type: gpuType, Size: int(p.GPUs)}
		if int(p.GPUs) != size {
			continue // width changed; this replica has to be re-placed
		}
		if err := fleet.Place(w, p.Node); err != nil {
			continue // node gone, cordoned, or now too full — re-place it
		}
		p.Domain = domainOf(fleet, p.Node)
		kept = append(kept, p)
	}

	// --- place whatever is still missing ---
	var pending []scheduler.Workload
	used := map[int32]bool{}
	for _, p := range kept {
		used[p.Replica] = true
	}
	for ordinal := int32(0); len(kept)+len(pending) < want; ordinal++ {
		if used[ordinal] {
			continue
		}
		pending = append(pending, scheduler.Workload{
			Name: replicaName(&ic, ordinal), Type: gpuType, Size: size,
		})
		used[ordinal] = true
	}

	topologyHonoured := true
	if len(pending) > 0 {
		if ic.Spec.RequireSameDomain {
			// Every replica of this cluster in one domain, including the ones
			// already placed — so this only works cleanly on a fresh placement.
			if len(kept) == 0 {
				if _, placed, ok := fleet.PackInOneDomain(pending, scheduler.BestFit); ok {
					kept = append(kept, toPlacements(&ic, placed, size, fleet)...)
					pending = nil
				} else {
					topologyHonoured = false
				}
			} else {
				topologyHonoured = false
			}
		}
		if len(pending) > 0 {
			placed, unplaced := fleet.Pack(pending, scheduler.BestFit)
			kept = append(kept, toPlacements(&ic, placed, size, fleet)...)
			if len(unplaced) > 0 {
				log.Info("short of capacity", "cluster", ic.Name,
					"unplaced", len(unplaced), "width", size,
					"fragmentation", fleet.Fragmentation(gpuType, size),
					"largestPlaceable", fleet.LargestPlaceable(gpuType))
			}
		}
	}

	sortPlacements(kept)

	// --- write status ---
	ready := int32(len(kept))
	prev := ic.Status.DeepCopy()

	ic.Status.ObservedGeneration = ic.Generation
	ic.Status.ReadyReplicas = ready
	ic.Status.Placements = kept
	ic.Status.Fragmentation = strconv.FormatFloat(fleet.Fragmentation(gpuType, size), 'f', 2, 64)
	ic.Status.LargestPlaceable = int32(fleet.LargestPlaceable(gpuType))

	switch {
	case want == 0:
		ic.Status.Phase = fleetv1.PhasePending
	case ready == int32(want):
		ic.Status.Phase = fleetv1.PhaseReady
	case ready > 0:
		ic.Status.Phase = fleetv1.PhaseDegraded
	default:
		ic.Status.Phase = fleetv1.PhasePending
	}

	setCond(&ic, fleetv1.ConditionPlaced, ready == int32(want),
		map[bool]string{true: "AllReplicasPlaced", false: "InsufficientCapacity"}[ready == int32(want)],
		fmt.Sprintf("%d/%d replicas placed", ready, want))

	canAdmit := fleet.LargestPlaceable(gpuType) >= size
	setCond(&ic, fleetv1.ConditionCapacity, canAdmit,
		map[bool]string{true: "CapacityAvailable", false: "Fragmented"}[canAdmit],
		fmt.Sprintf("widest placeable replica is %d GPUs, request is %d; fragmentation %s",
			fleet.LargestPlaceable(gpuType), size, ic.Status.Fragmentation))

	setCond(&ic, fleetv1.ConditionTopology, topologyHonoured,
		map[bool]string{true: "Satisfied", false: "CouldNotColocate"}[topologyHonoured],
		map[bool]string{
			true:  "replicas are within one interconnect domain, or locality was not requested",
			false: "no single interconnect domain could hold every replica",
		}[topologyHonoured])

	if !equalStatus(prev, &ic.Status) {
		if err := r.Status().Update(ctx, &ic); err != nil {
			if apierrors.IsConflict(err) {
				// Someone else wrote first. Re-read and redo rather than
				// forcing our view over theirs.
				return ctrl.Result{Requeue: true}, nil
			}
			return ctrl.Result{}, err
		}
	}

	if ic.Status.Phase == fleetv1.PhaseDegraded || (want > 0 && ready == 0) {
		return ctrl.Result{RequeueAfter: requeueDegraded}, nil
	}
	return ctrl.Result{}, nil
}

// drain releases capacity, then removes the finalizer so deletion completes.
// Releasing placement is the drain in this model; a real implementation evicts
// the serving pods first and waits for them to terminate, which is why this is
// a distinct phase and not just a finalizer removal.
func (r *InferenceClusterReconciler) drain(ctx context.Context, ic *fleetv1.InferenceCluster) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(ic, Finalizer) {
		return ctrl.Result{}, nil
	}
	if ic.Status.Phase != fleetv1.PhaseDraining || len(ic.Status.Placements) > 0 {
		ic.Status.Phase = fleetv1.PhaseDraining
		ic.Status.Placements = nil
		ic.Status.ReadyReplicas = 0
		ic.Status.ObservedGeneration = ic.Generation
		if err := r.Status().Update(ctx, ic); err != nil && !apierrors.IsConflict(err) {
			return ctrl.Result{}, err
		}
	}
	controllerutil.RemoveFinalizer(ic, Finalizer)
	if err := r.Update(ctx, ic); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// fleetFromNodes builds a scheduler view from the cluster's nodes.
//
// A node counts only if it advertises GPUs, carries a pool label, is not
// cordoned, and reports Ready. Anything else is capacity we must not schedule
// onto, and silently including a cordoned node is how a drain gets undone by
// the very controller that should respect it.
func (r *InferenceClusterReconciler) fleetFromNodes(ctx context.Context) (*scheduler.Fleet, error) {
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes); err != nil {
		return nil, err
	}
	var out []scheduler.Node
	for i := range nodes.Items {
		n := &nodes.Items[i]
		gpuType, ok := n.Labels[LabelGPUType]
		if !ok || gpuType == "" {
			continue
		}
		if n.Spec.Unschedulable {
			continue
		}
		if !nodeReady(n) {
			continue
		}
		q, ok := n.Status.Allocatable[ResourceGPU]
		if !ok {
			continue
		}
		count, ok := q.AsInt64()
		if !ok || count <= 0 {
			continue
		}
		out = append(out, scheduler.Node{
			Name:   n.Name,
			Type:   scheduler.GPUType(gpuType),
			Total:  int(count),
			Domain: n.Labels[LabelDomain],
		})
	}
	return scheduler.NewFleet(out)
}

func nodeReady(n *corev1.Node) bool {
	for _, c := range n.Status.Conditions {
		if c.Type == corev1.NodeReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}

// occupy replays a cluster's recorded placements into the fleet view, so the
// packer sees capacity that is already committed. Placements naming nodes that
// no longer exist are skipped: that cluster will re-place on its own reconcile,
// and blocking on it here would wedge every other cluster too.
func occupy(f *scheduler.Fleet, ic *fleetv1.InferenceCluster) {
	t := scheduler.GPUType(ic.Spec.GPUType)
	for _, p := range ic.Status.Placements {
		_ = f.Place(scheduler.Workload{
			Name: replicaName(ic, p.Replica), Type: t, Size: int(p.GPUs),
		}, p.Node)
	}
}

func replicaName(ic *fleetv1.InferenceCluster, ordinal int32) string {
	return fmt.Sprintf("%s/%s#%d", ic.Namespace, ic.Name, ordinal)
}

func ordinalOf(name string) int32 {
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '#' {
			v, err := strconv.Atoi(name[i+1:])
			if err != nil {
				return 0
			}
			return int32(v)
		}
	}
	return 0
}

func toPlacements(ic *fleetv1.InferenceCluster, placed []scheduler.Placement, size int, f *scheduler.Fleet) []fleetv1.Placement {
	out := make([]fleetv1.Placement, 0, len(placed))
	for _, p := range placed {
		out = append(out, fleetv1.Placement{
			Replica: ordinalOf(p.Workload),
			Node:    p.Node,
			Domain:  domainOf(f, p.Node),
			GPUs:    int32(size),
		})
	}
	return out
}

func domainOf(f *scheduler.Fleet, node string) string { return f.DomainOf(node) }

func sortPlacements(ps []fleetv1.Placement) {
	for i := 1; i < len(ps); i++ {
		for j := i; j > 0 && ps[j].Replica < ps[j-1].Replica; j-- {
			ps[j], ps[j-1] = ps[j-1], ps[j]
		}
	}
}

func setCond(ic *fleetv1.InferenceCluster, condType string, ok bool, reason, msg string) {
	status := metav1.ConditionFalse
	if ok {
		status = metav1.ConditionTrue
	}
	meta.SetStatusCondition(&ic.Status.Conditions, metav1.Condition{
		Type:               condType,
		Status:             status,
		Reason:             reason,
		Message:            msg,
		ObservedGeneration: ic.Generation,
	})
}

// equalStatus compares the parts we compute, so an unchanged reconcile does not
// issue a write. Controllers that update status unconditionally generate a watch
// event, which wakes themselves up, forever.
func equalStatus(a, b *fleetv1.InferenceClusterStatus) bool {
	if a.Phase != b.Phase || a.ObservedGeneration != b.ObservedGeneration ||
		a.ReadyReplicas != b.ReadyReplicas || a.Fragmentation != b.Fragmentation ||
		a.LargestPlaceable != b.LargestPlaceable || len(a.Placements) != len(b.Placements) {
		return false
	}
	for i := range a.Placements {
		if a.Placements[i] != b.Placements[i] {
			return false
		}
	}
	for _, want := range b.Conditions {
		got := meta.FindStatusCondition(a.Conditions, want.Type)
		if got == nil || got.Status != want.Status || got.Reason != want.Reason ||
			got.Message != want.Message || got.ObservedGeneration != want.ObservedGeneration {
			return false
		}
	}
	return true
}

// SetupWithManager wires the controller up. It watches nodes as well as
// InferenceClusters: a node going away, being cordoned, or joining the pool
// changes what is placeable, and finding that out on a 30-second poll instead
// of an event is the difference between a control plane and a cron job.
func (r *InferenceClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&fleetv1.InferenceCluster{}).
		Watches(&corev1.Node{}, r.enqueueAllClusters()).
		Named("inferencecluster").
		Complete(r)
}

// enqueueAllClusters maps any node event onto every InferenceCluster. That is
// deliberately broad: a node joining or leaving changes what is placeable for
// all of them, and there is no cheaper mapping that is also correct. The
// reconcile is a no-op write-wise when nothing changed, which is what makes the
// fan-out affordable.
func (r *InferenceClusterReconciler) enqueueAllClusters() handler.EventHandler {
	return handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, _ client.Object) []reconcile.Request {
		var list fleetv1.InferenceClusterList
		if err := r.List(ctx, &list); err != nil {
			logf.FromContext(ctx).Error(err, "listing clusters after a node event")
			return nil
		}
		out := make([]reconcile.Request, 0, len(list.Items))
		for i := range list.Items {
			out = append(out, reconcile.Request{NamespacedName: types.NamespacedName{
				Namespace: list.Items[i].Namespace,
				Name:      list.Items[i].Name,
			}})
		}
		return out
	})
}
