package neuron

import (
	"math"
	"testing"
)

func TestSoloNeuron(t *testing.T) {
	threshold := 0.01
	cin := make(chan float64)
	cout := make(chan float64)
	n := Neuron{Input: cin, Output: cout}

	var table = []struct {
		in          float64
		expectedOut float64
	}{
		{0, 0.5},
		{10, 1},
		{1000000, 1},
		{-10, 0},
		{-1000000, 0},
	}

	for _, row := range table {
		go n.Fire()
		cin <- float64(row.in)
		actual := <-cout

		diff := math.Abs(actual - row.expectedOut)
		if diff > threshold {
			t.Fatalf("Sigmoid function failed for i = %v with %v", row.in, actual)
		} else {
			t.Logf("S(%v) = %.4f", row.in, actual)
		}
	}
}
