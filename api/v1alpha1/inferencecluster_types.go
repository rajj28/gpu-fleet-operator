// Package v1alpha1 is the declarative surface the inference team talks to.
//
// The whole point of this API is that a caller says what they need — a model,
// a replica count, a GPU width — and never learns which hardware pool, serving
// runtime or scheduler satisfied it. Everything about placement, provisioning
// and repair lives behind the controller.
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// InferenceClusterSpec is what the caller declares. Nothing in here names a
// node, a runtime version or a scheduler: those are the platform's business.
type InferenceClusterSpec struct {
	// Model is the model to serve.
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// GPUType selects the hardware pool (for example "h100", "a100"). A replica
	// is never placed on a node of a different type, however much room it has.
	// +kubebuilder:validation:MinLength=1
	GPUType string `json:"gpuType"`

	// Replicas is how many serving replicas are wanted.
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// GPUsPerReplica is how many GPUs one replica needs on a single node.
	// +kubebuilder:validation:Minimum=1
	GPUsPerReplica int32 `json:"gpusPerReplica"`

	// RequireSameDomain asks for every replica inside one interconnect domain
	// (an NVLink island, a rack, a fabric partition). Collective operations
	// across replicas get the fast path; without it they may cross domains.
	// The controller reports Degraded rather than silently spreading if this
	// cannot be satisfied.
	// +optional
	RequireSameDomain bool `json:"requireSameDomain,omitempty"`

	// Runtime optionally pins the serving stack. Leave it empty and the
	// platform picks — which is the behaviour we want callers to rely on.
	// +optional
	Runtime string `json:"runtime,omitempty"`
}

// Phase is a coarse, human-readable rollup of status. Conditions carry the
// detail; this exists so `kubectl get` is useful.
// +kubebuilder:validation:Enum=Pending;Provisioning;Ready;Degraded;Draining;Failed
type Phase string

const (
	// PhasePending — accepted, not yet placed.
	PhasePending Phase = "Pending"
	// PhaseProvisioning — placed, replicas coming up.
	PhaseProvisioning Phase = "Provisioning"
	// PhaseReady — every replica placed and serving.
	PhaseReady Phase = "Ready"
	// PhaseDegraded — some replicas placed, the rest cannot be, usually
	// because free capacity is fragmented below the requested width.
	PhaseDegraded Phase = "Degraded"
	// PhaseDraining — being torn down; replicas evicted before the object goes.
	PhaseDraining Phase = "Draining"
	// PhaseFailed — the request cannot be satisfied as written.
	PhaseFailed Phase = "Failed"
)

// Condition types this controller sets.
const (
	// ConditionPlaced is true when every requested replica has a node.
	ConditionPlaced = "Placed"
	// ConditionCapacity is true when the fleet can still admit a replica of
	// the requested width. False with reason Fragmented is the interesting
	// case: there is free capacity, just not in usable shapes.
	ConditionCapacity = "Capacity"
	// ConditionTopology is true when RequireSameDomain was honoured, or was
	// not asked for.
	ConditionTopology = "Topology"
)

// Placement records where one replica landed. Written by the controller, read
// by anything that wants to know without asking the scheduler again.
type Placement struct {
	// Replica is the replica ordinal.
	Replica int32 `json:"replica"`
	// Node is the host it sits on.
	Node string `json:"node"`
	// Domain is that node's interconnect domain.
	// +optional
	Domain string `json:"domain,omitempty"`
	// GPUs held by this replica.
	GPUs int32 `json:"gpus"`
}

// InferenceClusterStatus is the observed state.
type InferenceClusterStatus struct {
	// +optional
	Phase Phase `json:"phase,omitempty"`

	// ObservedGeneration is the .metadata.generation this status was computed
	// from. If it trails .metadata.generation, the spec changed and this status
	// describes the old one — which is how a caller (or a human) tells "not
	// converged yet" from "converged, and this is the answer".
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Placements is the current assignment, ordered by replica.
	// +optional
	Placements []Placement `json:"placements,omitempty"`

	// Fragmentation is the share of free GPUs in this pool that cannot accept
	// a replica of the requested width, as a string so it round-trips exactly.
	// +optional
	Fragmentation string `json:"fragmentation,omitempty"`

	// LargestPlaceable is the widest replica the pool could admit right now.
	// +optional
	LargestPlaceable int32 `json:"largestPlaceable,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=ic
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.model`
// +kubebuilder:printcolumn:name="GPU",type=string,JSONPath=`.spec.gpuType`
// +kubebuilder:printcolumn:name="Want",type=integer,JSONPath=`.spec.replicas`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Frag",type=string,JSONPath=`.status.fragmentation`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// InferenceCluster is one declared serving deployment.
type InferenceCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceClusterSpec   `json:"spec,omitempty"`
	Status InferenceClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// InferenceClusterList is a list of InferenceClusters.
type InferenceClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferenceCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferenceCluster{}, &InferenceClusterList{})
}

// TotalGPUs is the GPUs this cluster wants in total.
func (s InferenceClusterSpec) TotalGPUs() int32 { return s.Replicas * s.GPUsPerReplica }
