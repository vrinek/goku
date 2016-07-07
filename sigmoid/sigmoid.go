package sigmoid

import "math"

// Sigmoid Function as defined by https://en.wikipedia.org/wiki/Sigmoid_function
func Sigmoid(t float64) float64 {
	return 1 / (1 + math.Pow(math.E, -t))
}
