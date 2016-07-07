package neuron

import (
	"math"
	"testing"
)

func TestNoErrorNewLayer(t *testing.T) {
	inputs := []<-chan float64{make(chan float64)}
	outputs := []chan<- float64{make(chan float64)}
	_, err := NewLayer(inputs, outputs)

	if err != nil {
		t.Fatal("Expected no error")
	}
}

func TestInputsErrorNewLayer(t *testing.T) {
	inputs := []<-chan float64{}
	outputs := []chan<- float64{make(chan float64)}
	_, err := NewLayer(inputs, outputs)

	if err == nil {
		t.Fatal("Expected an error when no inputs given")
	}
}

func TestOutputsErrorNewLayer(t *testing.T) {
	inputs := []<-chan float64{make(chan float64)}
	outputs := []chan<- float64{}
	_, err := NewLayer(inputs, outputs)

	if err == nil {
		t.Fatal("Expected an error when no outputs given")
	}
}

func TestNewLayerZeroWeight(t *testing.T) {
	cin := make(chan float64)
	cout := make(chan float64)
	l, err := NewLayer([]<-chan float64{cin}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("No error expected, got:", err)
	}

	go l.Fire()
	cin <- 1
	actual := <-cout
	expected := 0.5
	diff := math.Abs(actual - expected)
	if diff > 0.001 {
		t.Fatalf("Expected %v, got %.4f", expected, actual)
	}
}

func TestNewLayerNegativeWeight(t *testing.T) {
	cin := make(chan float64)
	cout := make(chan float64)
	l, err := NewLayer([]<-chan float64{cin}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("No error expected, got:", err)
	}

	l.SetWeight(0, 0, -10)

	go l.Fire()
	cin <- 1
	actual := <-cout
	expected := 0.0
	diff := math.Abs(actual - expected)
	if diff > 0.001 {
		t.Fatalf("Expected %v, got %.4f", expected, actual)
	}
}

func TestNewLayerPositiveWeight(t *testing.T) {
	cin := make(chan float64)
	cout := make(chan float64)
	l, err := NewLayer([]<-chan float64{cin}, []chan<- float64{cout})
	if err != nil {
		t.Fatal("No error expected, got:", err)
	}

	l.SetWeight(0, 0, 10)

	go l.Fire()
	cin <- 1
	actual := <-cout
	expected := 1.0
	diff := math.Abs(actual - expected)
	if diff > 0.001 {
		t.Log(l.Inspect())
		t.Fatalf("Expected %v, got %.4f", expected, actual)
	}
}
