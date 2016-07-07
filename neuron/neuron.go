package neuron

import "github.com/vrinek/goku/sigmoid"

type Neuron struct {
	Input  <-chan float64
	Output chan<- float64
}

func (n Neuron) Fire() {
	n.Output <- sigmoid.Sigmoid(<-n.Input)
}
