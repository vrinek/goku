package neuron

type Pathway struct {
	Weight float64
	Input  <-chan float64
	Output chan<- float64
}

func (p Pathway) Fire() {
	p.Output <- p.Weight * <-p.Input
}
