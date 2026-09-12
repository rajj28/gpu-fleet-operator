package scheduler

import (
	"fmt"
	"sort"
)

// Move is one workload relocation the defragmenter wants to make. Cost is the
// GPUs that must be drained and rescheduled, which is a usable proxy for how
// much disruption the move buys.
type Move struct {
	Workload string
	From     string
	To       string
	Cost     int
}

func (m Move) String() string {
	return fmt.Sprintf("%s: %s -> %s (%d GPU)", m.Workload, m.From, m.To, m.Cost)
}

// Plan is a proposed defragmentation, along with the before/after numbers that
// justify it. Nothing is applied until Apply is called — a control plane should
// be able to compute a plan, log it, and have a human or a policy decide.
type Plan struct {
	Moves []Move
	// NodesFreed is how many nodes end up holding nothing.
	NodesFreed int
	// Cost is total GPUs relocated.
	Cost int

	FragmentationBefore float64
	FragmentationAfter  float64
	LargestBefore       int
	LargestAfter        int
	TargetSize          int
	Type                GPUType
}

// Worthwhile reports whether the plan actually buys anything. A plan that
// relocates workloads without improving the widest placeable job is churn, and
// the caller should drop it.
func (p Plan) Worthwhile() bool {
	return len(p.Moves) > 0 && p.LargestAfter > p.LargestBefore
}

func (p Plan) String() string {
	return fmt.Sprintf(
		"defrag %s size=%d: %d moves, %d GPU relocated, %d nodes freed, fragmentation %.2f -> %.2f, largest %d -> %d",
		p.Type, p.TargetSize, len(p.Moves), p.Cost, p.NodesFreed,
		p.FragmentationBefore, p.FragmentationAfter, p.LargestBefore, p.LargestAfter)
}

// Defragment computes a plan that consolidates scattered workloads so the fleet
// can admit jobs `targetSize` GPUs wide.
//
// Strategy: evacuate the emptiest nodes first. A node holding one 2-GPU replica
// is cheap to clear and yields a whole node; a node holding six is expensive and
// yields the same one node. So candidates are sorted by how little they hold,
// and each is only evacuated if every workload on it can be rehomed elsewhere —
// a partial evacuation is strictly worse than none, because it pays the
// disruption and still leaves the node occupied.
//
// Rehoming uses best-fit onto *other* nodes, and never onto another node we are
// also trying to clear.
func (f *Fleet) Defragment(t GPUType, targetSize int) (Plan, error) {
	if targetSize <= 0 {
		return Plan{}, fmt.Errorf("target size must be positive, got %d", targetSize)
	}

	plan := Plan{
		Type:                t,
		TargetSize:          targetSize,
		FragmentationBefore: f.Fragmentation(t, targetSize),
		LargestBefore:       f.LargestPlaceable(t),
	}
	plan.FragmentationAfter = plan.FragmentationBefore
	plan.LargestAfter = plan.LargestBefore

	// If a target-size job already fits somewhere, we are done. Relocating
	// workloads to free up a *second* wide node is not free — it drains and
	// reschedules live traffic — and the fleet can already admit the shape we
	// were asked about. Without this guard the planner happily evacuates a
	// fully-packed node onto the empty one and calls it progress.
	if plan.LargestBefore >= targetSize {
		return plan, nil
	}

	// Work on a copy so a rejected plan leaves the real fleet untouched.
	sim, err := f.clone()
	if err != nil {
		return Plan{}, err
	}

	type candidate struct {
		node string
		held int
	}
	var candidates []candidate
	for _, name := range sim.order {
		n := sim.nodes[name]
		if n.Type != t || n.Total == 0 {
			continue
		}
		held := sim.used[name]
		free := sim.Free(name)
		// Skip three kinds of node:
		//   - empty: nothing to evacuate, nothing to gain.
		//   - already wide enough: it can take a target-size job as it stands.
		//   - fully allocated: it is contributing zero free GPUs, so it is not
		//     fragmenting anything. Clearing it is the most expensive move
		//     available and buys only what an empty node would have given.
		// Fragmentation lives on *partially* used nodes, and those are the only
		// ones worth touching.
		if held == 0 || free == 0 || free >= targetSize {
			continue
		}
		candidates = append(candidates, candidate{name, held})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].held != candidates[j].held {
			return candidates[i].held < candidates[j].held
		}
		return candidates[i].node < candidates[j].node
	})

	cleared := map[string]bool{}
	for _, c := range candidates {
		moves, ok := sim.tryEvacuate(c.node, cleared)
		if !ok {
			continue // could not fully clear it; leave it alone
		}
		cleared[c.node] = true
		plan.Moves = append(plan.Moves, moves...)
		for _, m := range moves {
			plan.Cost += m.Cost
		}
	}

	plan.FragmentationAfter = sim.Fragmentation(t, targetSize)
	plan.LargestAfter = sim.LargestPlaceable(t)
	plan.NodesFreed = len(cleared)
	return plan, nil
}

// tryEvacuate attempts to move everything off `node`. It either returns the full
// set of moves, or reports failure having changed nothing.
func (f *Fleet) tryEvacuate(node string, offLimits map[string]bool) ([]Move, bool) {
	var residents []string
	for w, n := range f.where {
		if n == node {
			residents = append(residents, w)
		}
	}
	sort.Slice(residents, func(i, j int) bool {
		if f.sizes[residents[i]] != f.sizes[residents[j]] {
			return f.sizes[residents[i]] > f.sizes[residents[j]] // widest first
		}
		return residents[i] < residents[j]
	})

	snapshot, err := f.clone()
	if err != nil {
		return nil, false
	}

	nodeType := f.nodes[node].Type
	var moves []Move
	for _, w := range residents {
		size := f.sizes[w]
		f.Evict(w)

		target := ""
		bestLeftover := int(^uint(0) >> 1)
		for _, cand := range f.order {
			if cand == node || offLimits[cand] {
				continue
			}
			if f.nodes[cand].Type != nodeType {
				continue
			}
			fr := f.Free(cand)
			if fr < size {
				continue
			}
			if leftover := fr - size; leftover < bestLeftover {
				target, bestLeftover = cand, leftover
			}
		}

		if target == "" {
			f.restore(snapshot)
			return nil, false
		}
		if err := f.Place(Workload{Name: w, Type: nodeType, Size: size}, target); err != nil {
			f.restore(snapshot)
			return nil, false
		}
		moves = append(moves, Move{Workload: w, From: node, To: target, Cost: size})
	}
	return moves, true
}

// Apply executes a plan against the fleet. It is all-or-nothing: if any move
// fails, the fleet is rolled back to where it started and the error is returned.
// A control plane that applies half a plan has made the fleet worse than when
// it started, so partial application is never the right answer.
func (f *Fleet) Apply(p Plan) error {
	snapshot, err := f.clone()
	if err != nil {
		return err
	}
	for i, m := range p.Moves {
		if got := f.NodeOf(m.Workload); got != m.From {
			f.restore(snapshot)
			return fmt.Errorf("move %d: %q is on %q, plan expected %q (fleet drifted since planning)",
				i, m.Workload, got, m.From)
		}
		size := f.sizes[m.Workload]
		t := f.nodes[m.From].Type
		f.Evict(m.Workload)
		if err := f.Place(Workload{Name: m.Workload, Type: t, Size: size}, m.To); err != nil {
			f.restore(snapshot)
			return fmt.Errorf("move %d (%s): %w", i, m, err)
		}
	}
	return nil
}

func (f *Fleet) clone() (*Fleet, error) {
	nodes := make([]Node, 0, len(f.order))
	for _, name := range f.order {
		nodes = append(nodes, f.nodes[name])
	}
	c, err := NewFleet(nodes)
	if err != nil {
		return nil, err
	}
	for w, n := range f.where {
		c.where[w] = n
		c.sizes[w] = f.sizes[w]
	}
	for n, u := range f.used {
		c.used[n] = u
	}
	return c, nil
}

func (f *Fleet) restore(s *Fleet) {
	f.where = make(map[string]string, len(s.where))
	f.sizes = make(map[string]int, len(s.sizes))
	f.used = make(map[string]int, len(s.used))
	for k, v := range s.where {
		f.where[k] = v
	}
	for k, v := range s.sizes {
		f.sizes[k] = v
	}
	for k, v := range s.used {
		f.used[k] = v
	}
}
