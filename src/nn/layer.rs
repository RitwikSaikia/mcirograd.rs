use rand::Rng;

use crate::{Module, Neuron, Value};

pub struct Layer {
  neurons: Vec<Neuron>,
}

impl Layer {
  pub fn new<R>(
    num_inputs: usize,
    num_outputs: usize,
    nonlin: bool,
    rng: &mut R,
  ) -> Self
  where
    R: Rng,
  {
    let neurons = (0..num_outputs)
      .map(|_| Neuron::new(num_inputs, nonlin, rng))
      .collect();

    Self { neurons }
  }
}

impl Module for Layer {
  fn parameters(&self) -> Vec<Value> {
    let mut result = vec![];
    for neuron in self.neurons.iter() {
      result.extend(neuron.parameters());
    }
    result
  }

  fn call(&self, inputs: &[&Value]) -> Vec<Value> {
    let mut result = vec![];
    for neuron in self.neurons.iter() {
      result.extend(neuron.call(inputs));
    }
    result
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::{SeedableRng, rngs::StdRng};

  #[test]
  fn test_layer() {
    let seed = [0u8; 32];
    let mut rng = StdRng::from_seed(seed);

    let layer = Layer::new(3, 2, true, &mut rng);

    let x0 = Value::new(2.0, Some("x0"));
    let x1 = Value::new(3.0, Some("x1"));
    let x2 = Value::new(-1.0, Some("x2"));

    let inputs = &[&x0, &x1, &x2];

    let outputs = layer.call(inputs);

    assert_eq!(2, outputs.len());

    let mut output = Value::new(0.0, None);
    for o in outputs.iter() {
      output += o;
    }
    output.set_label("output");

    output.backward();

    assert_eq!(output.data(), 1.9964161578120942);
    assert_eq!(x0.grad(), 0.003682978341454416);
    assert_eq!(x1.grad(), 0.004846327009305783);
    assert_eq!(x2.grad(), 0.0006329487522748086);

    let output_svg = output.into_svg();
    std::fs::write("/tmp/micrograd_layer.svg", output_svg).unwrap();
  }
}
