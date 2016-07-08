package sigmoid

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	threshold := 0.001

	var table = []struct {
		in          float64
		expectedOut float64
	}{
		{0, 0.5},
		{2, 0.88},
		{4, 0.982},
		{10, 1},
		{1000000, 1},
		{-2, 0.119},
		{-10, 0},
		{-1000000, 0},
	}

	for _, row := range table {
		actual := Sigmoid(row.in)

		diff := math.Abs(actual - row.expectedOut)
		if diff > threshold {
			t.Fatalf("Sigmoid function failed for i = %v with %v", row.in, actual)
		} else {
			t.Logf("S(%v) = %.4f", row.in, actual)
		}
	}
}
