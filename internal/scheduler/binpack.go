// Package scheduler packs GPU workloads onto fleet nodes and measures how much
// usable capacity the current placement is wasting.
//
// The problem this solves: a fleet can report 30% free GPUs and still be unable
// to admit a single 8-GPU job, because that 30% is scattered one or two GPUs at
// a time across forty hosts. Utilization alone does not surface that. The
// fragmentation metric below does, and Defragment turns it into a plan.
package scheduler

import (
	"fmt"
	"sort"
)

// GPUType distinguishes hardware pools. A workload is never placed on a node of
// a different type, no matter how much room that node has.
type GPUType string

// Node is one physical host in the fleet.
type Node struct {
	Name string
	Type GPUType
	// Total GPUs physically present.
	Total int
	// Domain is the interconnect domain (NVLink island, rack, or fabric
	// partition). Multi-GPU workloads placed inside one domain get the fast
	// path; split across domains they fall back to the slow one, so the packer
	// never splits a workload across domains.
	Domain string
}

// Workload is one placement unit: a model replica, a serving deployment, a
// training job. Size is how many GPUs it needs on a single node.
type Workload struct {
	Name string
	Type GPUType
	Size int
}

// Placement is the result of packing one workload.
type Placement struct {
	Workload string
	Node     string
}

// Fleet is a mutable view of nodes and what is currently placed on them.
type Fleet struct {
	nodes map[string]Node
	order []string // stable iteration order, so packing is deterministic
	used  map[string]int
	where map[string]string // workload name -> node name
	sizes map[string]int    // workload name -> GPUs held
}

// NewFleet builds a fleet view. Node names must be unique.
func NewFleet(nodes []Node) (*Fleet, error) {
	f := &Fleet{
		nodes: make(map[string]Node, len(nodes)),
		used:  make(map[string]int, len(nodes)),
		where: make(map[string]string),
		sizes: make(map[string]int),
	}
	for _, n := range nodes {
		if _, dup := f.nodes[n.Name]; dup {
			return nil, fmt.Errorf("duplicate node %q", n.Name)
		}
		if n.Total < 0 {
			return nil, fmt.Errorf("node %q has negative GPU count", n.Name)
		}
		f.nodes[n.Name] = n
		f.order = append(f.order, n.Name)
	}
	sort.Strings(f.order)
	return f, nil
}

// Free returns unallocated GPUs on a node.
func (f *Fleet) Free(node string) int {
	return f.nodes[node].Total - f.used[node]
}

// TotalFree returns unallocated GPUs across the fleet.
func (f *Fleet) TotalFree() int {
	sum := 0
	for _, name := range f.order {
		sum += f.Free(name)
	}
	return sum
}

// TotalCapacity returns all GPUs in the fleet.
func (f *Fleet) TotalCapacity() int {
	sum := 0
	for _, name := range f.order {
		sum += f.nodes[name].Total
	}
	return sum
}

// Utilization is allocated GPUs over total GPUs, in [0,1].
func (f *Fleet) Utilization() float64 {
	total := f.TotalCapacity()
	if total == 0 {
		return 0
	}
	return float64(total-f.TotalFree()) / float64(total)
}

// Place assigns a workload to a specific node. It is idempotent for an
// identical re-placement and rejects anything that would overcommit a node or
// cross a GPU-type boundary — the reconcile loop calls this on every pass, so
// it has to be safe to call with state it has already applied.
func (f *Fleet) Place(w Workload, node string) error {
	n, ok := f.nodes[node]
	if !ok {
		return fmt.Errorf("unknown node %q", node)
	}
	if existing, placed := f.where[w.Name]; placed {
		if existing == node {
			return nil // already where we want it
		}
		return fmt.Errorf("workload %q already placed on %q", w.Name, existing)
	}
	if w.Size <= 0 {
		return fmt.Errorf("workload %q must request at least one GPU", w.Name)
	}
	if n.Type != w.Type {
		return fmt.Errorf("workload %q wants %s, node %q is %s", w.Name, w.Type, node, n.Type)
	}
	if f.Free(node) < w.Size {
		return fmt.Errorf("node %q has %d free, workload %q needs %d", node, f.Free(node), w.Name, w.Size)
	}
	f.used[node] += w.Size
	f.where[w.Name] = node
	f.sizes[w.Name] = w.Size
	return nil
}

// Evict removes a workload. Unknown workloads are not an error, so a
// reconciliation that runs twice does not fail the second time.
func (f *Fleet) Evict(workload string) {
	node, ok := f.where[workload]
	if !ok {
		return
	}
	f.used[node] -= f.sizes[workload]
	delete(f.where, workload)
	delete(f.sizes, workload)
}

// NodeOf reports where a workload sits, or "" if unplaced.
func (f *Fleet) NodeOf(workload string) string { return f.where[workload] }

// Nodes returns node names in deterministic order.
func (f *Fleet) Nodes() []string {
	out := make([]string, len(f.order))
	copy(out, f.order)
	return out
}

// Workloads returns placed workload names in deterministic order.
func (f *Fleet) Workloads() []string {
	out := make([]string, 0, len(f.where))
	for w := range f.where {
		out = append(out, w)
	}
	sort.Strings(out)
	return out
}

// EmptyNodes counts nodes holding nothing. These are the ones that can take a
// full-width job, be power-capped, or be handed back to a different pool — which
// is the entire point of defragmenting.
func (f *Fleet) EmptyNodes() int {
	count := 0
	for _, name := range f.order {
		if f.used[name] == 0 && f.nodes[name].Total > 0 {
			count++
		}
	}
	return count
}

// Fragmentation reports the share of free GPUs that cannot accept a workload of
// the given size, for the given GPU type, because they sit on nodes with too
// little room left.
//
// Returns a value in [0,1]. Zero means every free GPU is usable for a job that
// size. One means none of them are — the fleet looks like it has capacity and
// cannot actually schedule anything that wide.
//
// This is the number `Utilization` hides. A fleet at 70% utilization with
// fragmentation 0.95 for size 8 has no 8-GPU capacity at all.
func (f *Fleet) Fragmentation(t GPUType, size int) float64 {
	if size <= 0 {
		return 0
	}
	var free, usable int
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type != t {
			continue
		}
		fr := f.Free(name)
		free += fr
		// Only whole multiples of `size` on this node are usable for jobs
		// this wide; a node with 3 free GPUs contributes nothing to 8-GPU demand.
		usable += (fr / size) * size
	}
	if free == 0 {
		return 0
	}
	return 1 - float64(usable)/float64(free)
}

// LargestPlaceable returns the widest workload of this type the fleet could
// admit right now.
func (f *Fleet) LargestPlaceable(t GPUType) int {
	best := 0
	for _, name := range f.order {
		if f.nodes[name].Type != t {
			continue
		}
		if fr := f.Free(name); fr > best {
			best = fr
		}
	}
	return best
}

// Strategy selects how a node is chosen for each workload.
type Strategy int

const (
	// FirstFitDecreasing places the widest workloads first onto the first node
	// with room. Fast, and good enough when demand is uniform.
	FirstFitDecreasing Strategy = iota
	// BestFit places each workload on the node that will have the least room
	// left afterwards, which keeps large holes intact for large jobs. Slower,
	// and noticeably better at holding fragmentation down under mixed demand.
	BestFit
)

// Pack places as many workloads as it can and reports what did not fit.
// Workloads are sorted widest-first regardless of strategy: placing a 1-GPU job
// before an 8-GPU job is how a fleet fragments in the first place.
func (f *Fleet) Pack(workloads []Workload, s Strategy) (placed []Placement, unplaced []Workload) {
	queue := make([]Workload, len(workloads))
	copy(queue, workloads)
	sort.SliceStable(queue, func(i, j int) bool {
		if queue[i].Size != queue[j].Size {
			return queue[i].Size > queue[j].Size
		}
		return queue[i].Name < queue[j].Name
	})

	for _, w := range queue {
		node := f.pick(w, s)
		if node == "" {
			unplaced = append(unplaced, w)
			continue
		}
		if err := f.Place(w, node); err != nil {
			unplaced = append(unplaced, w)
			continue
		}
		placed = append(placed, Placement{Workload: w.Name, Node: node})
	}
	return placed, unplaced
}

func (f *Fleet) pick(w Workload, s Strategy) string {
	best, bestLeftover := "", int(^uint(0)>>1)
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type != w.Type {
			continue
		}
		fr := f.Free(name)
		if fr < w.Size {
			continue
		}
		if s == FirstFitDecreasing {
			return name
		}
		if leftover := fr - w.Size; leftover < bestLeftover {
			best, bestLeftover = name, leftover
		}
	}
	return best
}
