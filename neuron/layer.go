package neuron

import (
	"errors"
	"fmt"
)

// Firer - TODO
type Firer interface {
	Fire()
}

// Layer - TODO
type Layer struct {
	inputs        []<-chan float64
	outputs       []chan<- float64
	elements      []*Firer
	pathwayLayers [][]*Pathway
}

// Fire - TODO
func (l Layer) Fire() {
	for _, e := range l.elements {
		go (*e).Fire()
	}
}

func (l *Layer) addElement(f Firer) {
	l.elements = append(l.elements, &f)
}

// SetWeight - TODO
func (l *Layer) SetWeight(i int, o int, weight float64) {
	(*l.pathwayLayers[i][o]).Weight = weight
}

// Inspect - TODO
func (l *Layer) Inspect() string {
	inspection := ""
	inspection += "Layer inputs\n"
	for i, input := range l.inputs {
		inspection += fmt.Sprintf("\t%v) %#v\n", i, input)
	}
	inspection += "Layer elements\n"
	for i, elem := range l.elements {
		inspection += fmt.Sprintf("\t%v) %#v\n", i, *elem)
	}
	inspection += "Layer outputs\n"
	for i, output := range l.outputs {
		inspection += fmt.Sprintf("\t%v) %#v\n", i, output)
	}
	return inspection
}

// NewLayer - TODO
func NewLayer(inputs []<-chan float64, outputs []chan<- float64) (Layer, error) {
	l := Layer{}

	if len(inputs) == 0 {
		return l, errors.New("A Layer should have at least one input")
	}

	if len(outputs) == 0 {
		return l, errors.New("A Layer should have at least one output")
	}

	l.inputs = inputs
	l.outputs = outputs
	l.pathwayLayers = make([][]*Pathway, len(inputs))

	// Keeps all in & out channels related to pathways accessible
	var pathIns = make([][]chan float64, len(inputs))   // [i][o]
	var pathOuts = make([][]chan float64, len(outputs)) // [o][i]
	for i := range inputs {
		pathIns[i] = make([]chan float64, len(outputs))
		for o := range outputs {
			pathIns[i][o] = make(chan float64)
		}
	}
	for o := range outputs {
		pathOuts[o] = make([]chan float64, len(inputs))
		for i := range inputs {
			pathOuts[o][i] = make(chan float64)
		}
	}

	for i, input := range inputs {
		r := OutgoingRouter{Input: input}
		l.addElement(&r)
		l.pathwayLayers[i] = make([]*Pathway, len(pathIns[i]))
		for o, pathIn := range pathIns[i] {
			r.Outputs = append(r.Outputs, pathIn)
			p := Pathway{Input: pathIn, Output: pathOuts[o][i]}
			l.pathwayLayers[i][o] = &p
			l.addElement(&p)
		}
	}
	for o, output := range outputs {
		rn := make(chan float64)
		r := IncomingRouter{Output: rn}
		n := Neuron{Input: rn, Output: output}
		l.addElement(&r)
		l.addElement(&n)
		for _, pathOut := range pathOuts[o] {
			r.Inputs = append(r.Inputs, pathOut)
		}
	}

	return l, nil
}
