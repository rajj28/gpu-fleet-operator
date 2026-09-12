package scheduler

import (
	"math"
	"testing"
)

const h100 GPUType = "h100"
const a100 GPUType = "a100"

func nodes8(names ...string) []Node {
	out := make([]Node, 0, len(names))
	for _, n := range names {
		out = append(out, Node{Name: n, Type: h100, Total: 8, Domain: "d1"})
	}
	return out
}

func mustFleet(t *testing.T, ns []Node) *Fleet {
	t.Helper()
	f, err := NewFleet(ns)
	if err != nil {
		t.Fatalf("NewFleet: %v", err)
	}
	return f
}

func TestNewFleetRejectsDuplicates(t *testing.T) {
	_, err := NewFleet([]Node{{Name: "a", Total: 8}, {Name: "a", Total: 8}})
	if err == nil {
		t.Fatal("expected duplicate node name to be rejected")
	}
}

func TestPlaceIsIdempotent(t *testing.T) {
	f := mustFleet(t, nodes8("n1"))
	w := Workload{Name: "w1", Type: h100, Size: 4}

	if err := f.Place(w, "n1"); err != nil {
		t.Fatalf("first place: %v", err)
	}
	// The reconcile loop calls Place on every pass. Re-placing a workload
	// where it already is must be a no-op, not a double-charge.
	if err := f.Place(w, "n1"); err != nil {
		t.Fatalf("second place should be a no-op, got %v", err)
	}
	if got := f.Free("n1"); got != 4 {
		t.Fatalf("free = %d, want 4 (idempotent place double-charged the node)", got)
	}
}

func TestPlaceRejectsOvercommitAndTypeMismatch(t *testing.T) {
	f := mustFleet(t, []Node{
		{Name: "h", Type: h100, Total: 8},
		{Name: "a", Type: a100, Total: 8},
	})
	if err := f.Place(Workload{Name: "big", Type: h100, Size: 9}, "h"); err == nil {
		t.Error("expected overcommit to be rejected")
	}
	if err := f.Place(Workload{Name: "wrong", Type: h100, Size: 2}, "a"); err == nil {
		t.Error("expected GPU-type mismatch to be rejected")
	}
	if err := f.Place(Workload{Name: "zero", Type: h100, Size: 0}, "h"); err == nil {
		t.Error("expected zero-GPU workload to be rejected")
	}
}

func TestEvictUnknownWorkloadIsSafe(t *testing.T) {
	f := mustFleet(t, nodes8("n1"))
	f.Evict("never-existed") // must not panic or corrupt accounting
	if got := f.Free("n1"); got != 8 {
		t.Fatalf("free = %d, want 8", got)
	}
}

// The headline case: a fleet that reports plenty of free capacity and cannot
// schedule a single wide job.
func TestFragmentationExposesUnusableCapacity(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2", "n3", "n4"))
	// Six GPUs used on each of four 8-GPU nodes -> 8 free fleet-wide, spread
	// two at a time. Utilization looks healthy at 75%.
	for _, n := range f.Nodes() {
		if err := f.Place(Workload{Name: "load-" + n, Type: h100, Size: 6}, n); err != nil {
			t.Fatalf("place on %s: %v", n, err)
		}
	}

	if got := f.TotalFree(); got != 8 {
		t.Fatalf("TotalFree = %d, want 8", got)
	}
	if got := f.Utilization(); math.Abs(got-0.75) > 1e-9 {
		t.Fatalf("Utilization = %v, want 0.75", got)
	}

	// Eight GPUs free, and not one 8-GPU job can land.
	if got := f.Fragmentation(h100, 8); got != 1.0 {
		t.Errorf("Fragmentation(size 8) = %v, want 1.0 — 8 free GPUs, none usable", got)
	}
	if got := f.LargestPlaceable(h100); got != 2 {
		t.Errorf("LargestPlaceable = %d, want 2", got)
	}
	// The same free capacity is perfectly usable for 2-GPU work.
	if got := f.Fragmentation(h100, 2); got != 0.0 {
		t.Errorf("Fragmentation(size 2) = %v, want 0.0", got)
	}
	// Odd sizes waste the remainder: 2 free per node, size 3 -> nothing usable.
	if got := f.Fragmentation(h100, 3); got != 1.0 {
		t.Errorf("Fragmentation(size 3) = %v, want 1.0", got)
	}
}

func TestFragmentationEmptyFleetIsZero(t *testing.T) {
	f := mustFleet(t, nodes8("n1"))
	if err := f.Place(Workload{Name: "full", Type: h100, Size: 8}, "n1"); err != nil {
		t.Fatal(err)
	}
	// No free GPUs at all: fragmentation is undefined-but-zero, not NaN.
	if got := f.Fragmentation(h100, 4); got != 0 {
		t.Fatalf("Fragmentation on a full fleet = %v, want 0", got)
	}
}

func TestPackPlacesWidestFirst(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2"))
	// Deliberately queued narrow-first. If the packer honoured this order with
	// first-fit it would scatter the small jobs and the 8-GPU job would not fit.
	work := []Workload{
		{Name: "small-a", Type: h100, Size: 1},
		{Name: "small-b", Type: h100, Size: 1},
		{Name: "wide", Type: h100, Size: 8},
	}
	placed, unplaced := f.Pack(work, BestFit)
	if len(unplaced) != 0 {
		t.Fatalf("unplaced = %v, want none (widest-first ordering should make all three fit)", unplaced)
	}
	if len(placed) != 3 {
		t.Fatalf("placed %d, want 3", len(placed))
	}
	if f.NodeOf("wide") == f.NodeOf("small-a") {
		t.Error("the 8-GPU workload shares a node with a 1-GPU workload, which is impossible on 8-GPU nodes")
	}
}

func TestPackReportsWhatDidNotFit(t *testing.T) {
	f := mustFleet(t, nodes8("n1"))
	_, unplaced := f.Pack([]Workload{
		{Name: "fits", Type: h100, Size: 8},
		{Name: "does-not", Type: h100, Size: 8},
		{Name: "wrong-type", Type: a100, Size: 1},
	}, BestFit)
	if len(unplaced) != 2 {
		t.Fatalf("unplaced = %d (%v), want 2", len(unplaced), unplaced)
	}
}

func TestBestFitKeepsLargeHolesIntact(t *testing.T) {
	// Two nodes: one with 4 free, one with 8 free. A 4-GPU job should take the
	// node that is already partly used, preserving the empty node for wide work.
	f := mustFleet(t, nodes8("n1", "n2"))
	if err := f.Place(Workload{Name: "sitting", Type: h100, Size: 4}, "n1"); err != nil {
		t.Fatal(err)
	}
	f.Pack([]Workload{{Name: "incoming", Type: h100, Size: 4}}, BestFit)

	if got := f.NodeOf("incoming"); got != "n1" {
		t.Errorf("best-fit put the 4-GPU job on %q; want n1, so n2 stays whole", got)
	}
	if got := f.LargestPlaceable(h100); got != 8 {
		t.Errorf("LargestPlaceable = %d, want 8 — best-fit failed to preserve the empty node", got)
	}
}

// Defragmentation: the scattered fleet from the fragmentation test, consolidated.
func TestDefragmentConsolidatesScatteredWorkloads(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2", "n3", "n4"))
	// Each node holds a single 2-GPU replica: 8 used, 24 free, and no 8-GPU job
	// can land because every node has 6 free, not 8.
	for _, n := range f.Nodes() {
		if err := f.Place(Workload{Name: "replica-" + n, Type: h100, Size: 2}, n); err != nil {
			t.Fatal(err)
		}
	}
	if got := f.LargestPlaceable(h100); got != 6 {
		t.Fatalf("precondition: LargestPlaceable = %d, want 6", got)
	}

	plan, err := f.Defragment(h100, 8)
	if err != nil {
		t.Fatalf("Defragment: %v", err)
	}
	if !plan.Worthwhile() {
		t.Fatalf("plan not worthwhile: %s", plan)
	}
	if plan.NodesFreed == 0 {
		t.Error("no nodes freed")
	}
	if plan.LargestAfter < 8 {
		t.Errorf("LargestAfter = %d, want >= 8 so an 8-GPU job fits", plan.LargestAfter)
	}
	if plan.FragmentationAfter >= plan.FragmentationBefore {
		t.Errorf("fragmentation did not improve: %.2f -> %.2f",
			plan.FragmentationBefore, plan.FragmentationAfter)
	}
	t.Logf("%s", plan)

	// Applying the plan must reproduce exactly what it promised.
	if err := f.Apply(plan); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if got := f.LargestPlaceable(h100); got != plan.LargestAfter {
		t.Errorf("after Apply LargestPlaceable = %d, plan promised %d", got, plan.LargestAfter)
	}
	// Nothing may be lost in the shuffle.
	if got := len(f.Workloads()); got != 4 {
		t.Errorf("%d workloads survived defrag, want 4", got)
	}
	if got := f.TotalFree(); got != 24 {
		t.Errorf("TotalFree = %d after defrag, want 24 — GPU accounting leaked", got)
	}
}

func TestDefragmentLeavesAlreadyPackedFleetAlone(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2"))
	if err := f.Place(Workload{Name: "full", Type: h100, Size: 8}, "n1"); err != nil {
		t.Fatal(err)
	}
	// n2 is empty, so an 8-GPU job already fits. There is nothing to do.
	plan, err := f.Defragment(h100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Moves) != 0 {
		t.Errorf("moves = %v, want none on an already-schedulable fleet", plan.Moves)
	}
	if plan.Worthwhile() {
		t.Error("plan reported worthwhile with nothing to gain")
	}
}

func TestDefragmentDoesNotPartiallyEvacuate(t *testing.T) {
	// One node holds 6 GPUs of work; nowhere else has room for it. The
	// defragmenter must not move half of it and leave the node occupied.
	f := mustFleet(t, nodes8("n1", "n2"))
	if err := f.Place(Workload{Name: "a", Type: h100, Size: 3}, "n1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "b", Type: h100, Size: 3}, "n1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "blocker", Type: h100, Size: 7}, "n2"); err != nil {
		t.Fatal(err)
	}

	before := f.NodeOf("a") + "/" + f.NodeOf("b")
	plan, err := f.Defragment(h100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Moves) != 0 {
		t.Errorf("moves = %v; n1 cannot be fully evacuated so it must be left alone", plan.Moves)
	}
	if after := f.NodeOf("a") + "/" + f.NodeOf("b"); after != before {
		t.Errorf("planning mutated the live fleet: %s -> %s", before, after)
	}
}

func TestDefragmentRejectsBadTargetSize(t *testing.T) {
	f := mustFleet(t, nodes8("n1"))
	if _, err := f.Defragment(h100, 0); err == nil {
		t.Error("expected target size 0 to be rejected")
	}
}

func TestApplyDetectsDriftSincePlanning(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2", "n3"))
	for _, n := range f.Nodes() {
		if err := f.Place(Workload{Name: "r-" + n, Type: h100, Size: 2}, n); err != nil {
			t.Fatal(err)
		}
	}
	plan, err := f.Defragment(h100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Moves) == 0 {
		t.Skip("no moves planned; nothing to drift against")
	}

	// Something else moved the workload between planning and applying —
	// exactly the race a reconcile loop hits in production.
	victim := plan.Moves[0].Workload
	f.Evict(victim)

	if err := f.Apply(plan); err == nil {
		t.Fatal("Apply accepted a plan whose premise no longer held")
	}
}

func TestApplyRollsBackOnFailure(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2"))
	if err := f.Place(Workload{Name: "w1", Type: h100, Size: 2}, "n1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "w2", Type: h100, Size: 2}, "n2"); err != nil {
		t.Fatal(err)
	}

	// A plan whose second move is impossible: n1 cannot take 8 more GPUs.
	bad := Plan{Moves: []Move{
		{Workload: "w1", From: "n1", To: "n2", Cost: 2},
		{Workload: "w2", From: "n2", To: "nonexistent", Cost: 2},
	}}
	if err := f.Apply(bad); err == nil {
		t.Fatal("expected Apply to fail")
	}
	// First move must have been undone.
	if got := f.NodeOf("w1"); got != "n1" {
		t.Errorf("w1 is on %q after rollback, want n1", got)
	}
	if got := f.NodeOf("w2"); got != "n2" {
		t.Errorf("w2 is on %q after rollback, want n2", got)
	}
}

func TestEmptyNodes(t *testing.T) {
	f := mustFleet(t, nodes8("n1", "n2", "n3"))
	if err := f.Place(Workload{Name: "w", Type: h100, Size: 1}, "n1"); err != nil {
		t.Fatal(err)
	}
	if got := f.EmptyNodes(); got != 2 {
		t.Fatalf("EmptyNodes = %d, want 2", got)
	}
}
