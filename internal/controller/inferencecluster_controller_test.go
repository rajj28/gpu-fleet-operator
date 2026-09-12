package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	fleetv1 "github.com/rajj28/gpu-fleet-operator/api/v1alpha1"
)

// These run against controller-runtime's fake client: no API server, no etcd,
// no envtest binaries to download. The fake client honours status subresources
// and finalizers, which is what these tests are actually about.

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := fleetv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	return s
}

func gpuNode(name, gpuType, domain string, gpus int64, ready bool, cordoned bool) *corev1.Node {
	cond := corev1.ConditionTrue
	if !ready {
		cond = corev1.ConditionFalse
	}
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{LabelGPUType: gpuType, LabelDomain: domain},
		},
		Spec: corev1.NodeSpec{Unschedulable: cordoned},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{ResourceGPU: *resource.NewQuantity(gpus, resource.DecimalSI)},
			Conditions:  []corev1.NodeCondition{{Type: corev1.NodeReady, Status: cond}},
		},
	}
}

func cluster(name string, replicas, gpusEach int32, sameDomain bool) *fleetv1.InferenceCluster {
	return &fleetv1.InferenceCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: "default", Generation: 1,
			Finalizers: []string{Finalizer},
		},
		Spec: fleetv1.InferenceClusterSpec{
			Model: "llama-3.3-70b", GPUType: "h100",
			Replicas: replicas, GPUsPerReplica: gpusEach,
			RequireSameDomain: sameDomain,
		},
	}
}

func newReconciler(t *testing.T, objs ...client.Object) (*InferenceClusterReconciler, client.Client) {
	t.Helper()
	s := testScheme(t)
	c := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(objs...).
		WithStatusSubresource(&fleetv1.InferenceCluster{}).
		Build()
	return &InferenceClusterReconciler{Client: c, Scheme: s}, c
}

func reconcileOnce(t *testing.T, r *InferenceClusterReconciler, name string) ctrl.Result {
	t.Helper()
	res, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: name},
	})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	return res
}

func load(t *testing.T, c client.Client, name string) *fleetv1.InferenceCluster {
	t.Helper()
	var ic fleetv1.InferenceCluster
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: name}, &ic); err != nil {
		t.Fatalf("get: %v", err)
	}
	return &ic
}

func TestPlacesEveryReplicaAndReportsReady(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		cluster("serve", 2, 8, false),
	)
	reconcileOnce(t, r, "serve")

	ic := load(t, c, "serve")
	if ic.Status.Phase != fleetv1.PhaseReady {
		t.Fatalf("phase = %s, want Ready", ic.Status.Phase)
	}
	if ic.Status.ReadyReplicas != 2 || len(ic.Status.Placements) != 2 {
		t.Fatalf("ready=%d placements=%d, want 2/2", ic.Status.ReadyReplicas, len(ic.Status.Placements))
	}
	if ic.Status.ObservedGeneration != ic.Generation {
		t.Errorf("ObservedGeneration = %d, want %d", ic.Status.ObservedGeneration, ic.Generation)
	}
	if a, b := ic.Status.Placements[0].Node, ic.Status.Placements[1].Node; a == b {
		t.Errorf("both 8-GPU replicas landed on %q, which holds 8 GPUs", a)
	}
	if !meta.IsStatusConditionTrue(ic.Status.Conditions, fleetv1.ConditionPlaced) {
		t.Error("Placed condition should be true")
	}
}

// The property that matters most: reconciling again must not move anything.
func TestReconcileIsStableAndDoesNotChurnPlacements(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		gpuNode("n3", "h100", "d1", 8, true, false),
		cluster("serve", 2, 4, false),
	)
	reconcileOnce(t, r, "serve")
	first := load(t, c, "serve").Status.Placements
	if len(first) != 2 {
		t.Fatalf("expected 2 placements, got %d", len(first))
	}

	for i := 0; i < 4; i++ {
		reconcileOnce(t, r, "serve")
		got := load(t, c, "serve").Status.Placements
		if len(got) != len(first) {
			t.Fatalf("pass %d: placement count changed %d -> %d", i, len(first), len(got))
		}
		for j := range got {
			if got[j] != first[j] {
				t.Fatalf("pass %d: replica %d migrated %v -> %v (churn)",
					i, got[j].Replica, first[j], got[j])
			}
		}
	}
}

// Fragmented capacity: free GPUs exist, none of them usable at the requested
// width. The cluster must report Degraded with a Fragmented reason rather than
// silently placing nothing and looking merely Pending.
func TestFragmentedFleetReportsDegradedNotSilentFailure(t *testing.T) {
	// Two nodes, each already carrying a 6-GPU replica from another cluster.
	other := cluster("incumbent", 2, 6, false)
	other.Name = "incumbent"
	other.Status.Placements = []fleetv1.Placement{
		{Replica: 0, Node: "n1", GPUs: 6},
		{Replica: 1, Node: "n2", GPUs: 6},
	}

	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		other,
		cluster("wide", 1, 8, false), // wants 8 contiguous; 4 free, 2+2
	)
	reconcileOnce(t, r, "wide")

	ic := load(t, c, "wide")
	if ic.Status.ReadyReplicas != 0 {
		t.Fatalf("ready = %d, want 0 — there is no 8-GPU hole", ic.Status.ReadyReplicas)
	}
	cond := meta.FindStatusCondition(ic.Status.Conditions, fleetv1.ConditionCapacity)
	if cond == nil {
		t.Fatal("Capacity condition missing")
	}
	if cond.Status != metav1.ConditionFalse || cond.Reason != "Fragmented" {
		t.Errorf("Capacity = %s/%s, want False/Fragmented", cond.Status, cond.Reason)
	}
	if ic.Status.Fragmentation != "1.00" {
		t.Errorf("fragmentation = %q, want \"1.00\"", ic.Status.Fragmentation)
	}
	if ic.Status.LargestPlaceable != 2 {
		t.Errorf("LargestPlaceable = %d, want 2", ic.Status.LargestPlaceable)
	}
}

func TestCordonedAndNotReadyNodesAreNotScheduledOnto(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("cordoned", "h100", "d1", 8, true, true),
		gpuNode("notready", "h100", "d1", 8, false, false),
		gpuNode("good", "h100", "d1", 8, true, false),
		cluster("serve", 3, 8, false),
	)
	reconcileOnce(t, r, "serve")

	ic := load(t, c, "serve")
	if ic.Status.ReadyReplicas != 1 {
		t.Fatalf("ready = %d, want 1 — only one node is schedulable", ic.Status.ReadyReplicas)
	}
	if got := ic.Status.Placements[0].Node; got != "good" {
		t.Errorf("placed on %q; cordoned and NotReady nodes must be excluded", got)
	}
	if ic.Status.Phase != fleetv1.PhaseDegraded {
		t.Errorf("phase = %s, want Degraded", ic.Status.Phase)
	}
}

func TestRequireSameDomainKeepsReplicasTogether(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("a1", "h100", "island-a", 8, true, false),
		gpuNode("a2", "h100", "island-a", 8, true, false),
		gpuNode("b1", "h100", "island-b", 8, true, false),
		gpuNode("b2", "h100", "island-b", 8, true, false),
		cluster("colocated", 2, 8, true),
	)
	reconcileOnce(t, r, "colocated")

	ic := load(t, c, "colocated")
	if ic.Status.ReadyReplicas != 2 {
		t.Fatalf("ready = %d, want 2", ic.Status.ReadyReplicas)
	}
	d0, d1 := ic.Status.Placements[0].Domain, ic.Status.Placements[1].Domain
	if d0 == "" || d0 != d1 {
		t.Errorf("domains %q and %q; RequireSameDomain means one island", d0, d1)
	}
	if !meta.IsStatusConditionTrue(ic.Status.Conditions, fleetv1.ConditionTopology) {
		t.Error("Topology condition should be true")
	}
}

func TestRequireSameDomainReportsWhenItCannotColocate(t *testing.T) {
	// Each domain has exactly one free node, so two replicas cannot share one.
	blocker := cluster("blocker", 2, 8, false)
	blocker.Name = "blocker"
	blocker.Status.Placements = []fleetv1.Placement{
		{Replica: 0, Node: "a1", GPUs: 8},
		{Replica: 1, Node: "b1", GPUs: 8},
	}
	r, c := newReconciler(t,
		gpuNode("a1", "h100", "island-a", 8, true, false),
		gpuNode("a2", "h100", "island-a", 8, true, false),
		gpuNode("b1", "h100", "island-b", 8, true, false),
		gpuNode("b2", "h100", "island-b", 8, true, false),
		blocker,
		cluster("colocated", 2, 8, true),
	)
	reconcileOnce(t, r, "colocated")

	ic := load(t, c, "colocated")
	cond := meta.FindStatusCondition(ic.Status.Conditions, fleetv1.ConditionTopology)
	if cond == nil || cond.Status != metav1.ConditionFalse {
		t.Fatalf("Topology condition = %v, want False", cond)
	}
	if cond.Reason != "CouldNotColocate" {
		t.Errorf("reason = %q, want CouldNotColocate", cond.Reason)
	}
	// It still places what it can rather than refusing outright.
	if ic.Status.ReadyReplicas == 0 {
		t.Error("expected fallback placement across domains, got nothing")
	}
}

func TestScaleUpKeepsExistingReplicasPut(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		cluster("serve", 1, 4, false),
	)
	reconcileOnce(t, r, "serve")
	before := load(t, c, "serve").Status.Placements[0]

	ic := load(t, c, "serve")
	ic.Spec.Replicas = 3
	ic.Generation = 2
	if err := c.Update(context.Background(), ic); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(t, r, "serve")

	after := load(t, c, "serve")
	if after.Status.ReadyReplicas != 3 {
		t.Fatalf("ready = %d, want 3", after.Status.ReadyReplicas)
	}
	if after.Status.Placements[0] != before {
		t.Errorf("replica 0 moved on scale-up: %v -> %v", before, after.Status.Placements[0])
	}
	if after.Status.ObservedGeneration != 2 {
		t.Errorf("ObservedGeneration = %d, want 2", after.Status.ObservedGeneration)
	}
}

func TestScaleDownReleasesCapacity(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		cluster("serve", 4, 4, false),
	)
	reconcileOnce(t, r, "serve")
	if got := load(t, c, "serve").Status.ReadyReplicas; got != 4 {
		t.Fatalf("setup: ready = %d, want 4", got)
	}

	ic := load(t, c, "serve")
	ic.Spec.Replicas = 1
	ic.Generation = 2
	if err := c.Update(context.Background(), ic); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(t, r, "serve")

	after := load(t, c, "serve")
	if after.Status.ReadyReplicas != 1 || len(after.Status.Placements) != 1 {
		t.Fatalf("ready=%d placements=%d, want 1/1", after.Status.ReadyReplicas, len(after.Status.Placements))
	}
	if after.Status.LargestPlaceable != 8 {
		t.Errorf("LargestPlaceable = %d after scale-down, want 8 — capacity was not released",
			after.Status.LargestPlaceable)
	}
}

// A replica whose node vanished must be re-placed, not silently lost.
func TestReplacesReplicaWhoseNodeDisappeared(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		cluster("serve", 1, 8, false),
	)
	reconcileOnce(t, r, "serve")
	ic := load(t, c, "serve")
	host := ic.Status.Placements[0].Node

	var node corev1.Node
	if err := c.Get(context.Background(), types.NamespacedName{Name: host}, &node); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete(context.Background(), &node); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(t, r, "serve")

	after := load(t, c, "serve")
	if after.Status.ReadyReplicas != 1 {
		t.Fatalf("ready = %d, want 1 — the replica should have moved to the surviving node", after.Status.ReadyReplicas)
	}
	if after.Status.Placements[0].Node == host {
		t.Errorf("still placed on the deleted node %q", host)
	}
}

func TestDeletionDrainsThenRemovesFinalizer(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		cluster("serve", 1, 8, false),
	)
	reconcileOnce(t, r, "serve")

	ic := load(t, c, "serve")
	if err := c.Delete(context.Background(), ic); err != nil {
		t.Fatal(err)
	}
	reconcileOnce(t, r, "serve")

	var check fleetv1.InferenceCluster
	err := c.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "serve"}, &check)
	if err == nil && len(check.Finalizers) > 0 {
		t.Fatalf("finalizer %v still present after drain", check.Finalizers)
	}
}

func TestZeroReplicasIsPendingNotReady(t *testing.T) {
	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		cluster("idle", 0, 8, false),
	)
	reconcileOnce(t, r, "idle")

	ic := load(t, c, "idle")
	if ic.Status.Phase != fleetv1.PhasePending {
		t.Errorf("phase = %s, want Pending for a zero-replica cluster", ic.Status.Phase)
	}
	if len(ic.Status.Placements) != 0 {
		t.Errorf("placements = %v, want none", ic.Status.Placements)
	}
}

func TestOtherClustersCapacityIsRespected(t *testing.T) {
	incumbent := cluster("incumbent", 1, 8, false)
	incumbent.Name = "incumbent"
	incumbent.Status.Placements = []fleetv1.Placement{{Replica: 0, Node: "n1", GPUs: 8}}

	r, c := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		gpuNode("n2", "h100", "d1", 8, true, false),
		incumbent,
		cluster("newcomer", 2, 8, false),
	)
	reconcileOnce(t, r, "newcomer")

	ic := load(t, c, "newcomer")
	if ic.Status.ReadyReplicas != 1 {
		t.Fatalf("ready = %d, want 1 — n1 belongs to the incumbent", ic.Status.ReadyReplicas)
	}
	if got := ic.Status.Placements[0].Node; got != "n2" {
		t.Errorf("placed on %q, want n2; the incumbent's capacity was double-booked", got)
	}
}

func TestNodesWithoutGPULabelAreIgnored(t *testing.T) {
	plain := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "cpu-only"},
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}},
		},
	}
	r, c := newReconciler(t, plain, cluster("serve", 1, 1, false))
	reconcileOnce(t, r, "serve")

	ic := load(t, c, "serve")
	if ic.Status.ReadyReplicas != 0 {
		t.Errorf("ready = %d, want 0 — an unlabelled CPU node is not GPU capacity", ic.Status.ReadyReplicas)
	}
}

func TestDegradedClusterIsRequeued(t *testing.T) {
	r, _ := newReconciler(t,
		gpuNode("n1", "h100", "d1", 8, true, false),
		cluster("serve", 2, 8, false),
	)
	res := reconcileOnce(t, r, "serve")
	if res.RequeueAfter == 0 {
		t.Error("a degraded cluster should be requeued to retry when capacity frees up")
	}
}
