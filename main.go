package main

import (
	"fmt"

	"github.com/vrinek/goku/sigmoid"
)

func main() {
	for i := -10; i <= 10; i++ {
		s := sigmoid.Sigmoid(float64(i))

		fmt.Printf("Sigmoid(%v) = %.4f\n", i, s)
	}
}
