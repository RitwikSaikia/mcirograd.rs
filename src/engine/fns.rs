use crate::Value;

impl Value {
  pub fn relu(&self) -> Value {
    let input = self.data();
    let value = if input > 0.0 { input } else { 0.0 };
    let mut result = Value::new(value, None);
    result.set_op(Some("relu"));
    result.add_prev(&[self]);

    let mut tmp_self = self.clone();

    result.set_grad_fn(Box::new(move |grad| {
      let local_grad = if input > 0.0 { 1.0 } else { 0.0 };
      tmp_self.add_grad(grad * local_grad);
    }));

    result
  }

  pub fn tanh(&self) -> Value {
    let input = self.data();
    let value = input.tanh();
    let mut result = Value::new(value, None);
    result.set_op(Some("tanh"));
    result.add_prev(&[self]);

    let mut tmp_self = self.clone();

    result.set_grad_fn(Box::new(move |grad| {
      let local_grad = 1.0 - value * value;
      tmp_self.add_grad(grad * local_grad);
    }));

    result
  }
}

#[cfg(test)]
mod tests {
  use crate::Value;

  #[test]
  fn test_relu() {
    let x: Value = (-2.0).into();
    let mut relu_x = x.relu();
    assert_eq!(relu_x.data(), 0.0);
    relu_x.backward();
    assert_eq!(x.grad(), 0.0);

    let y: Value = 3.0.into();
    let mut relu_y = y.relu();
    assert_eq!(relu_y.data(), 3.0);
    relu_y.backward();
    assert_eq!(y.grad(), 1.0);
  }

  #[test]
  fn test_tanh() {
    let x: Value = (-2.0).into();
    let mut tanh_x = x.tanh();
    assert_eq!(tanh_x.data(), -0.9640275800758169);
    tanh_x.backward();
    assert_eq!(x.grad(), 0.07065082485316443);

    let y: Value = 3.0.into();
    let mut tanh_y = y.tanh();
    assert_eq!(tanh_y.data(), 0.9950547536867305);
    tanh_y.backward();
    assert_eq!(y.grad(), 0.009866037165440211);

    let z: Value = 0.0.into();
    let mut tanh_z = z.tanh();
    assert_eq!(tanh_z.data(), 0.0);
    tanh_z.backward();
    assert_eq!(z.grad(), 1.0);
  }
}
