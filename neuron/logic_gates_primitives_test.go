package neuron

import (
	"math"
	"testing"
)

func TestAndNetwork(t *testing.T) {
	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	p0r1 := make(chan float64)
	p1r1 := make(chan float64)
	p2r1 := make(chan float64)
	p0 := Pathway{Weight: -30, Input: c0, Output: p0r1}
	p1 := Pathway{Weight: 20, Input: c1, Output: p1r1}
	p2 := Pathway{Weight: 20, Input: c2, Output: p2r1}

	r1n1 := make(chan float64)
	r1 := IncomingRouter{
		Inputs: []<-chan float64{p0r1, p1r1, p2r1},
		Output: r1n1,
	}

	n1 := Neuron{Input: r1n1, Output: cout}

	elements := []Firer{p0, p1, p2, r1, n1}

	var expectations = []struct {
		a        float64
		b        float64
		expected float64
	}{
		{0, 0, 0},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 1},
	}
	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		c0 <- 1 // bias
		c1 <- expectation.a
		c2 <- expectation.b
		actual := <-cout

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("%v AND %v = %.4f", expectation.a, expectation.b, actual)
		}
	}
}

func TestOrNetwork(t *testing.T) {
	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	p0r1 := make(chan float64)
	p1r1 := make(chan float64)
	p2r1 := make(chan float64)
	p0 := Pathway{Weight: -10, Input: c0, Output: p0r1}
	p1 := Pathway{Weight: 20, Input: c1, Output: p1r1}
	p2 := Pathway{Weight: 20, Input: c2, Output: p2r1}

	r1n1 := make(chan float64)
	r1 := IncomingRouter{
		Inputs: []<-chan float64{p0r1, p1r1, p2r1},
		Output: r1n1,
	}

	n1 := Neuron{Input: r1n1, Output: cout}

	elements := []Firer{p0, p1, p2, r1, n1}

	var expectations = []struct {
		a        float64
		b        float64
		expected float64
	}{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 1},
	}
	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		c0 <- 1 // bias
		c1 <- expectation.a
		c2 <- expectation.b
		actual := <-cout

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("%v OR %v = %.4f", expectation.a, expectation.b, actual)
		}
	}
}

func TestNotNetwork(t *testing.T) {
	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	cout := make(chan float64)

	p0r1 := make(chan float64)
	p1r1 := make(chan float64)
	p0 := Pathway{Weight: 10, Input: c0, Output: p0r1}
	p1 := Pathway{Weight: -20, Input: c1, Output: p1r1}

	r1n1 := make(chan float64)
	r1 := IncomingRouter{
		Inputs: []<-chan float64{p0r1, p1r1},
		Output: r1n1,
	}

	n1 := Neuron{Input: r1n1, Output: cout}

	elements := []Firer{p0, p1, r1, n1}

	var expectations = []struct {
		a        float64
		expected float64
	}{
		{0, 1},
		{1, 0},
	}
	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		c0 <- 1 // bias
		c1 <- expectation.a
		actual := <-cout

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("NOT %v = %.4f", expectation.a, actual)
		}
	}
}

func TestNotAndNotNetwork(t *testing.T) {
	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	p0r1 := make(chan float64)
	p1r1 := make(chan float64)
	p2r1 := make(chan float64)
	p0 := Pathway{Weight: 10, Input: c0, Output: p0r1}
	p1 := Pathway{Weight: -20, Input: c1, Output: p1r1}
	p2 := Pathway{Weight: -20, Input: c2, Output: p2r1}

	r1n1 := make(chan float64)
	r1 := IncomingRouter{
		Inputs: []<-chan float64{p0r1, p1r1, p2r1},
		Output: r1n1,
	}

	n1 := Neuron{Input: r1n1, Output: cout}

	elements := []Firer{p0, p1, p2, r1, n1}

	var expectations = []struct {
		a        float64
		b        float64
		expected float64
	}{
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 0},
	}
	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		c0 <- 1 // bias
		c1 <- expectation.a
		c2 <- expectation.b
		actual := <-cout

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("%v OR %v = %.4f", expectation.a, expectation.b, actual)
		}
	}
}

func TestXnorNetwork(t *testing.T) {
	// XNOR can be rewritten as "AND OR (NOT AND NOT)"
	var expectations = []struct {
		a        float64
		b        float64
		expected float64
	}{
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 1},
	}

	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	// Input routing
	r0p0 := make(chan float64)
	r0p3 := make(chan float64)
	r0p6 := make(chan float64)
	r0 := OutgoingRouter{
		Input:   c0,
		Outputs: []chan<- float64{r0p0, r0p3, r0p6},
	}

	r1p1 := make(chan float64)
	r1p4 := make(chan float64)
	r1 := OutgoingRouter{
		Input:   c1,
		Outputs: []chan<- float64{r1p1, r1p4},
	}

	r2p2 := make(chan float64)
	r2p5 := make(chan float64)
	r2 := OutgoingRouter{
		Input:   c2,
		Outputs: []chan<- float64{r2p2, r2p5},
	}

	// AND
	p0r3 := make(chan float64) // bias
	p1r3 := make(chan float64)
	p2r3 := make(chan float64)
	p0 := Pathway{Weight: -30, Input: r0p0, Output: p0r3}
	p1 := Pathway{Weight: 20, Input: r1p1, Output: p1r3}
	p2 := Pathway{Weight: 20, Input: r2p2, Output: p2r3}

	r3n1 := make(chan float64)
	r3 := IncomingRouter{
		Inputs: []<-chan float64{p0r3, p1r3, p2r3},
		Output: r3n1,
	}

	n1p7 := make(chan float64)
	n1 := Neuron{Input: r3n1, Output: n1p7}

	// NOT AND NOT
	p3r4 := make(chan float64) // bias
	p4r4 := make(chan float64)
	p5r4 := make(chan float64)
	p3 := Pathway{Weight: 10, Input: r0p3, Output: p3r4}
	p4 := Pathway{Weight: -20, Input: r1p4, Output: p4r4}
	p5 := Pathway{Weight: -20, Input: r2p5, Output: p5r4}

	r4n2 := make(chan float64)
	r4 := IncomingRouter{
		Inputs: []<-chan float64{p3r4, p4r4, p5r4},
		Output: r4n2,
	}

	n2p8 := make(chan float64)
	n2 := Neuron{Input: r4n2, Output: n2p8}

	// OR
	p6r5 := make(chan float64) // bias
	p7r5 := make(chan float64)
	p8r5 := make(chan float64)
	p6 := Pathway{Weight: -10, Input: r0p6, Output: p6r5}
	p7 := Pathway{Weight: 20, Input: n1p7, Output: p7r5}
	p8 := Pathway{Weight: 20, Input: n2p8, Output: p8r5}

	r5n3 := make(chan float64)
	r5 := IncomingRouter{
		Inputs: []<-chan float64{p6r5, p7r5, p8r5},
		Output: r5n3,
	}

	n3 := Neuron{Input: r5n3, Output: cout}

	elements := []Firer{
		p0, p1, p2, p3, p4, p5, p6, p7, p8,
		r0, r1, r2, r3, r4, r5,
		n1, n2, n3,
	}

	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		c0 <- 1 // bias
		c1 <- expectation.a
		c2 <- expectation.b
		actual := <-cout

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("%v OR %v = %.4f", expectation.a, expectation.b, actual)
		}
	}
}
