package neuron

type IncomingRouter struct {
	Inputs []<-chan float64
	Output chan<- float64
}

func (r IncomingRouter) Fire() {
	acc := 0.0
	for _, input := range r.Inputs {
		acc += <-input
	}
	r.Output <- acc
}

type OutgoingRouter struct {
	Input   <-chan float64
	Outputs []chan<- float64
}

func (r OutgoingRouter) Fire() {
	val := <-r.Input
	for _, output := range r.Outputs {
		output <- val
	}
}
