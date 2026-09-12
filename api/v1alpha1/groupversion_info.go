package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

// GroupVersion is the group and version this API lives under.
var GroupVersion = schema.GroupVersion{Group: "fleet.gpu.io", Version: "v1alpha1"}

// SchemeBuilder registers the types in this package with a runtime.Scheme.
var SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

// AddToScheme adds these types to a Scheme.
var AddToScheme = SchemeBuilder.AddToScheme
