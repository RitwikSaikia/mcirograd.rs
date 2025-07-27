use crate::{Module, Value};
use rand::Rng;

pub struct Neuron {
  w: Vec<Value>,
  b: Value,
  nonlin: bool,
}

impl Neuron {
  pub fn new<R>(num_input: usize, nonlin: bool, rng: &mut R) -> Self
  where
    R: Rng,
  {
    let mut w = Vec::with_capacity(num_input);
    for i in 0..num_input {
      w.push(Value::new(rng.random(), Some(&format!("w{}", i))));
    }
    let b = Value::new(rng.random(), Some("b"));
    Self { w, b, nonlin }
  }
}

impl Module for Neuron {
  fn parameters(&self) -> Vec<Value> {
    let mut params = self.w.clone();
    params.push(self.b.clone());
    params
  }

  fn call(&self, inputs: &[&Value]) -> Vec<Value> {
    assert_eq!(inputs.len(), self.w.len());
    let mut act = self.b.clone();
    for (wi, &xi) in self.w.iter().zip(inputs.iter()) {
      act += wi * xi;
    }
    if self.nonlin {
      act = act.tanh();
    }
    vec![act]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::{SeedableRng, rngs::StdRng};

  use crate::{Module, Value};

  #[test]
  fn test_neuron() {
    let seed = [0u8; 32];
    let mut rng = StdRng::from_seed(seed);

    let neuron = Neuron::new(3, true, &mut rng);

    let x0 = Value::new(2.0, Some("x0"));
    let x1 = Value::new(3.0, Some("x1"));
    let x2 = Value::new(-1.0, Some("x2"));

    let inputs = vec![&x0, &x1, &x2];

    let outputs = neuron.call(&inputs);

    assert_eq!(1, outputs.len());

    let mut output = outputs[0].clone();
    output.set_label("output");

    output.backward();

    assert_eq!(output.data(), 0.997985090094488);
    assert_eq!(output.grad(), 1.0);

    assert_eq!(x0.grad(), 0.0013205428887850636);
    assert_eq!(x1.grad(), 0.003357614387079053);
    assert_eq!(x2.grad(), 0.00032169198856720486);

    let output_svg = output.into_svg();
    std::fs::write("/tmp/micrograd_neuron.svg", output_svg).unwrap();
  }
}
