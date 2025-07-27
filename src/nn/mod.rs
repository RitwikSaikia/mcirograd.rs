use crate::Value;
pub use layer::*;
pub use mlp::*;
pub use neuron::*;

mod layer;
mod mlp;
mod neuron;

pub trait Module {
  fn parameters(&self) -> Vec<Value>;

  fn zero_grad(&mut self) {
    for mut param in self.parameters() {
      param.zero_grad();
    }
  }

  fn call(&self, inputs: &[&Value]) -> Vec<Value>;
}
