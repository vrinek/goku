package neuron

import (
	"math"
	"testing"

	"github.com/vrinek/goku/neuron"
)

func TestAndLayerNetwork(t *testing.T) {
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

	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	l, err := neuron.NewLayer([]<-chan float64{c0, c1, c2}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("Expected no error")
	}

	l.SetWeight(0, 0, -30)
	l.SetWeight(1, 0, 20)
	l.SetWeight(2, 0, 20)

	for _, expectation := range expectations {
		go l.Fire()

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

func TestOrLayerNetwork(t *testing.T) {
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

	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	l, err := neuron.NewLayer([]<-chan float64{c0, c1, c2}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("Expected no error")
	}

	l.SetWeight(0, 0, -10)
	l.SetWeight(1, 0, 20)
	l.SetWeight(2, 0, 20)

	for _, expectation := range expectations {
		go l.Fire()

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

func TestNotLayerNetwork(t *testing.T) {
	var expectations = []struct {
		a        float64
		expected float64
	}{
		{0, 1},
		{1, 0},
	}

	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	cout := make(chan float64)

	l, err := neuron.NewLayer([]<-chan float64{c0, c1}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("Expected no error")
	}

	l.SetWeight(0, 0, 10)
	l.SetWeight(1, 0, -20)

	for _, expectation := range expectations {
		go l.Fire()

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

func TestNotAndNotLayerNetwork(t *testing.T) {
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

	c0 := make(chan float64) // bias
	c1 := make(chan float64)
	c2 := make(chan float64)
	cout := make(chan float64)

	l, err := neuron.NewLayer([]<-chan float64{c0, c1, c2}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("Expected no error")
	}

	l.SetWeight(0, 0, 10)
	l.SetWeight(1, 0, -20)
	l.SetWeight(2, 0, -20)

	for _, expectation := range expectations {
		go l.Fire()

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

func TestXnorLayersNetwork(t *testing.T) {
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

	in0l1 := make(chan float64) // bias
	in1l1 := make(chan float64)
	in2l1 := make(chan float64)

	in0l2 := make(chan float64) // bias
	in1l2 := make(chan float64)
	in2l2 := make(chan float64)

	out := make(chan float64)

	// Layer 1
	l1, err := neuron.NewLayer(
		[]<-chan float64{in0l1, in1l1, in2l1},
		[]chan<- float64{in1l2, in2l2},
	)
	if err != nil {
		t.Fatal("Expected no error")
	}

	// AND
	l1.SetWeight(0, 0, -30)
	l1.SetWeight(1, 0, 20)
	l1.SetWeight(2, 0, 20)

	// NOT AND NOT
	l1.SetWeight(0, 1, 10)
	l1.SetWeight(1, 1, -20)
	l1.SetWeight(2, 1, -20)

	// Layer 2
	l2, err := neuron.NewLayer(
		[]<-chan float64{in0l2, in1l2, in2l2},
		[]chan<- float64{out},
	)
	if err != nil {
		t.Fatal("Expected no error")
	}

	// OR
	l2.SetWeight(0, 0, -10)
	l2.SetWeight(1, 0, 20)
	l2.SetWeight(2, 0, 20)

	elements := []neuron.Firer{l1, l2}

	for _, expectation := range expectations {
		for _, element := range elements {
			go element.Fire()
		}

		in0l1 <- 1 // bias
		in0l2 <- 1 // bias
		in1l1 <- expectation.a
		in2l1 <- expectation.b
		actual := <-out

		diff := math.Abs(actual - expectation.expected)
		if diff > 0.001 {
			t.Fatalf("Expected %v, got %v", expectation.expected, actual)
		} else {
			t.Logf("%v OR %v = %.4f", expectation.a, expectation.b, actual)
		}
	}
}
