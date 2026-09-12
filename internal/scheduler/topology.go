package scheduler

import "sort"

// Interconnect topology. A multi-replica deployment doing collective operations
// across replicas wants them inside one domain — an NVLink island, a rack, a
// fabric partition. Crossing a domain boundary does not fail, it just drops to
// the slow path, which is the kind of regression nobody notices until someone
// benchmarks it.

// Domains lists the distinct interconnect domains holding nodes of this GPU
// type, sorted for determinism.
func (f *Fleet) Domains(t GPUType) []string {
	seen := map[string]bool{}
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type == t {
			seen[n.Domain] = true
		}
	}
	out := make([]string, 0, len(seen))
	for d := range seen {
		out = append(out, d)
	}
	sort.Strings(out)
	return out
}

// FreeInDomain returns unallocated GPUs of this type inside one domain.
func (f *Fleet) FreeInDomain(t GPUType, domain string) int {
	sum := 0
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type == t && n.Domain == domain {
			sum += f.Free(name)
		}
	}
	return sum
}

// FragmentationInDomain is Fragmentation, scoped to one domain. A fleet can look
// fine globally and still be unable to serve a domain-local request.
func (f *Fleet) FragmentationInDomain(t GPUType, domain string, size int) float64 {
	if size <= 0 {
		return 0
	}
	var free, usable int
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type != t || n.Domain != domain {
			continue
		}
		fr := f.Free(name)
		free += fr
		usable += (fr / size) * size
	}
	if free == 0 {
		return 0
	}
	return 1 - float64(usable)/float64(free)
}

// PackInOneDomain places every workload inside a single interconnect domain.
//
// Domains are tried richest-first. It is all-or-nothing per domain: if a domain
// cannot hold the whole set, nothing is placed there and the next is tried. If
// no single domain fits, it places nothing and reports ok=false — a half
// domain-local placement gives the caller neither the locality they asked for
// nor the capacity, so whether to spread instead is their decision, not this
// function's.
func (f *Fleet) PackInOneDomain(workloads []Workload, s Strategy) (domain string, placed []Placement, ok bool) {
	if len(workloads) == 0 {
		return "", nil, true
	}
	t := workloads[0].Type
	need := 0
	for _, w := range workloads {
		if w.Type != t {
			return "", nil, false // a domain-local set must be one hardware pool
		}
		need += w.Size
	}

	candidates := f.Domains(t)
	sort.SliceStable(candidates, func(i, j int) bool {
		fi, fj := f.FreeInDomain(t, candidates[i]), f.FreeInDomain(t, candidates[j])
		if fi != fj {
			return fi > fj
		}
		return candidates[i] < candidates[j]
	})

	for _, d := range candidates {
		if f.FreeInDomain(t, d) < need {
			continue // cheap reject before doing the work
		}
		sim, err := f.clone()
		if err != nil {
			return "", nil, false
		}
		sim.restrictTo(t, d)
		got, unplaced := sim.Pack(workloads, s)
		if len(unplaced) > 0 {
			continue // enough GPUs in aggregate, wrong shapes
		}
		// Commit against the real fleet on the nodes the simulation chose.
		for _, p := range got {
			var w Workload
			for _, cand := range workloads {
				if cand.Name == p.Workload {
					w = cand
					break
				}
			}
			if err := f.Place(w, p.Node); err != nil {
				return "", nil, false
			}
		}
		return d, got, true
	}
	return "", nil, false
}

// restrictTo drops every node outside one (type, domain) from this view. Only
// called on a clone.
func (f *Fleet) restrictTo(t GPUType, domain string) {
	keep := make([]string, 0, len(f.order))
	for _, name := range f.order {
		n := f.nodes[name]
		if n.Type == t && n.Domain == domain {
			keep = append(keep, name)
			continue
		}
		delete(f.nodes, name)
		delete(f.used, name)
	}
	f.order = keep
}

// DomainOf returns a node's interconnect domain, or "" if the node is unknown.
func (f *Fleet) DomainOf(node string) string { return f.nodes[node].Domain }

// TypeOf returns a node's GPU type, or "" if the node is unknown.
func (f *Fleet) TypeOf(node string) GPUType { return f.nodes[node].Type }
