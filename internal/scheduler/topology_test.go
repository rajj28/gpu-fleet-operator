package scheduler

import "testing"

func twoDomains() []Node {
	return []Node{
		{Name: "a1", Type: h100, Total: 8, Domain: "island-a"},
		{Name: "a2", Type: h100, Total: 8, Domain: "island-a"},
		{Name: "b1", Type: h100, Total: 8, Domain: "island-b"},
		{Name: "b2", Type: h100, Total: 8, Domain: "island-b"},
	}
}

func TestDomainsAndFreeInDomain(t *testing.T) {
	f := mustFleet(t, twoDomains())
	got := f.Domains(h100)
	if len(got) != 2 || got[0] != "island-a" || got[1] != "island-b" {
		t.Fatalf("Domains = %v, want [island-a island-b] sorted", got)
	}
	if n := f.FreeInDomain(h100, "island-a"); n != 16 {
		t.Fatalf("FreeInDomain(island-a) = %d, want 16", n)
	}
	if n := f.FreeInDomain(h100, "nonexistent"); n != 0 {
		t.Fatalf("FreeInDomain(nonexistent) = %d, want 0", n)
	}
}

func TestPackInOneDomainKeepsReplicasTogether(t *testing.T) {
	f := mustFleet(t, twoDomains())
	// Two 8-GPU replicas. Spread across domains they would fit trivially; the
	// point is that they must not be.
	work := []Workload{
		{Name: "r0", Type: h100, Size: 8},
		{Name: "r1", Type: h100, Size: 8},
	}
	domain, placed, ok := f.PackInOneDomain(work, BestFit)
	if !ok {
		t.Fatal("PackInOneDomain failed on an empty two-domain fleet")
	}
	if len(placed) != 2 {
		t.Fatalf("placed %d, want 2", len(placed))
	}
	for _, p := range placed {
		if d := domainOf(f, p.Node); d != domain {
			t.Errorf("replica %s landed in %q, expected all in %q", p.Workload, d, domain)
		}
	}
}

func TestPackInOneDomainRefusesToSpread(t *testing.T) {
	f := mustFleet(t, twoDomains())
	// Fill one node in each domain so neither domain can hold three 8-GPU
	// replicas, though the fleet as a whole has room for them.
	if err := f.Place(Workload{Name: "sitting-a", Type: h100, Size: 8}, "a1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "sitting-b", Type: h100, Size: 8}, "b1"); err != nil {
		t.Fatal(err)
	}
	if f.TotalFree() != 16 {
		t.Fatalf("precondition: TotalFree = %d, want 16", f.TotalFree())
	}

	work := []Workload{
		{Name: "r0", Type: h100, Size: 8},
		{Name: "r1", Type: h100, Size: 8},
		{Name: "r2", Type: h100, Size: 8},
	}
	_, placed, ok := f.PackInOneDomain(work, BestFit)
	if ok {
		t.Fatalf("expected failure: no single domain holds 24 GPUs (placed %v)", placed)
	}
	if len(placed) != 0 {
		t.Errorf("placed %v on failure; must be all-or-nothing", placed)
	}
	// And it must not have half-placed anything against the live fleet.
	if f.TotalFree() != 16 {
		t.Errorf("TotalFree = %d after a failed domain pack, want 16 untouched", f.TotalFree())
	}
}

func TestPackInOneDomainPicksRichestDomain(t *testing.T) {
	nodes := append(twoDomains(), Node{Name: "c1", Type: h100, Total: 8, Domain: "island-c"})
	f := mustFleet(t, nodes)
	// Leave island-b with the least room.
	if err := f.Place(Workload{Name: "fill-b1", Type: h100, Size: 8}, "b1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "fill-b2", Type: h100, Size: 6}, "b2"); err != nil {
		t.Fatal(err)
	}

	domain, _, ok := f.PackInOneDomain([]Workload{{Name: "r0", Type: h100, Size: 2}}, BestFit)
	if !ok {
		t.Fatal("pack failed")
	}
	if domain == "island-b" {
		t.Errorf("chose the poorest domain %q; richest-first should avoid it", domain)
	}
}

func TestFragmentationInDomainIsScoped(t *testing.T) {
	f := mustFleet(t, twoDomains())
	// island-a scattered: 6 used on each of its two nodes -> 4 free, 2 per node.
	if err := f.Place(Workload{Name: "x1", Type: h100, Size: 6}, "a1"); err != nil {
		t.Fatal(err)
	}
	if err := f.Place(Workload{Name: "x2", Type: h100, Size: 6}, "a2"); err != nil {
		t.Fatal(err)
	}
	if got := f.FragmentationInDomain(h100, "island-a", 8); got != 1.0 {
		t.Errorf("island-a fragmentation(8) = %v, want 1.0", got)
	}
	// island-b is untouched and perfectly able to take an 8-GPU replica.
	if got := f.FragmentationInDomain(h100, "island-b", 8); got != 0.0 {
		t.Errorf("island-b fragmentation(8) = %v, want 0.0", got)
	}
	// Globally it looks like only half the free capacity is unusable, which is
	// exactly why the per-domain number is the one a domain-local request needs.
	if got := f.Fragmentation(h100, 8); got == 1.0 {
		t.Errorf("global fragmentation = %v; expected it to understate the domain-local problem", got)
	}
}

func TestPackInOneDomainRejectsMixedTypes(t *testing.T) {
	f := mustFleet(t, twoDomains())
	_, _, ok := f.PackInOneDomain([]Workload{
		{Name: "r0", Type: h100, Size: 2},
		{Name: "r1", Type: a100, Size: 2},
	}, BestFit)
	if ok {
		t.Error("a domain-local set spanning two hardware pools should be rejected")
	}
}

func TestPackInOneDomainEmptySetIsOK(t *testing.T) {
	f := mustFleet(t, twoDomains())
	if _, _, ok := f.PackInOneDomain(nil, BestFit); !ok {
		t.Error("empty workload set should succeed trivially")
	}
}

func domainOf(f *Fleet, node string) string { return f.nodes[node].Domain }
