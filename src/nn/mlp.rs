use rand::Rng;

use crate::{Layer, Module, Value};

pub struct MLP {
  layers: Vec<Layer>,
}

impl MLP {
  pub fn new<R>(num_inputs: usize, hidden_layers: &[usize], rng: &mut R) -> Self
  where
    R: Rng,
  {
    let mut sizes = Vec::with_capacity(hidden_layers.len() + 1);
    sizes.push(num_inputs);
    sizes.extend_from_slice(hidden_layers);

    let mut layers = Vec::with_capacity(hidden_layers.len() + 1);

    for i in 0..hidden_layers.len() {
      layers.push(Layer::new(
        sizes[i],
        sizes[i + 1],
        i != hidden_layers.len(),
        rng,
      ));
    }

    Self { layers }
  }
}

impl Module for MLP {
  fn parameters(&self) -> Vec<Value> {
    let mut result = vec![];
    for layer in &self.layers {
      result.extend(layer.parameters());
    }
    result
  }

  fn call(&self, inputs: &[&Value]) -> Vec<Value> {
    let mut outputs = inputs.iter().cloned().cloned().collect::<Vec<_>>();

    for layer in &self.layers {
      let tmp = outputs.iter().collect::<Vec<_>>();
      outputs = layer.call(tmp.as_slice());
    }

    outputs
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

    let mlp = MLP::new(3, &[4, 4, 1], &mut rng);

    let x0 = Value::new(2.0, Some("x0"));
    let x1 = Value::new(3.0, Some("x1"));
    let x2 = Value::new(-1.0, Some("x2"));

    let inputs = &[&x0, &x1, &x2];

    let outputs = mlp.call(inputs);

    assert_eq!(1, outputs.len());

    let mut output = outputs[0].clone();
    output.set_label("output");

    output.backward();

    assert_eq!(output.data(), 0.9743957547369949);
    assert_eq!(x0.grad(), 0.00039792547965581055);
    assert_eq!(x1.grad(), 0.0012243473508472232);
    assert_eq!(x2.grad(), 0.0034410038652130017);

    let output_svg = output.into_svg();
    std::fs::write("/tmp/micrograd_mlp.svg", output_svg).unwrap();
  }

  #[test]
  fn test_mlp_training() {
    let mut rng = StdRng::from_seed([0u8; 32]);
    let mut mlp = MLP::new(3, &[4, 4, 1], &mut rng);

    let xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    .iter()
    .map(|x| x.iter().map(|&v| Value::new(v, None)).collect::<Vec<_>>())
    .collect::<Vec<_>>();

    let ys = [1.0, -1.0, -1.0, 1.0]
      .iter()
      .map(|&v| Value::new(v, None))
      .collect::<Vec<_>>();

    let mut last_loss = f64::MAX;

    for _ in 0..10_000 {
      let ypred: Vec<_> = xs
        .iter()
        .map(|xi| {
          let xi: Vec<_> = xi.iter().map(|v| v as &Value).collect();
          mlp.call(&xi)[0].clone()
        })
        .collect();

      let mut loss = ypred
        .iter()
        .zip(&ys)
        .map(|(yp, yt)| (yp - yt))
        .map(|l| &l * &l)
        .fold(Value::from(0), |acc, l| acc + l);

      mlp.zero_grad();
      loss.backward();

      for mut p in mlp.parameters() {
        p.set_data(p.data() - 0.1 * p.grad());
      }

      last_loss = last_loss.min(loss.data());
      if loss.data() < 0.001 {
        break;
      }
    }

    let predict = |x: &[Value]| {
      let x: Vec<_> = x.iter().map(|v| v as &Value).collect();
      mlp.call(&x)[0].data()
    };

    assert_eq!(last_loss, 0.000998836703327823);
    assert_eq!(predict(&xs[0]), 0.9839666336902276);
    assert_eq!(predict(&xs[1]), -0.9879060572138844);
    assert_eq!(predict(&xs[2]), -0.9829406649047894);
    assert_eq!(predict(&xs[3]), 0.9826812630335173);
  }
}
