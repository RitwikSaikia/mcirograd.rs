use std::ops::*;

use crate::Value;

macro_rules! impl_op_ref {
  ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident) => {
    impl $Op<&Value> for Value {
      type Output = Value;

      fn $op(self, rhs: &Value) -> Self::Output {
        self.$op(rhs.clone())
      }
    }

    impl $Op<Value> for &Value {
      type Output = Value;

      fn $op(self, rhs: Value) -> Self::Output {
        self.clone().$op(rhs)
      }
    }

    impl $Op<&Value> for &Value {
      type Output = Value;

      fn $op(self, rhs: &Value) -> Self::Output {
        self.clone().$op(rhs.clone())
      }
    }

    impl $OpAssign<&Value> for Value {
      fn $op_assign(&mut self, rhs: &Value) {
        let result = self.clone().$op(rhs.clone());
        let result = result.inner();
        self.set_inner(result);
      }
    }
  };
}

macro_rules! impl_op_num {
  ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $ty:ty) => {
    impl $Op<$ty> for Value {
      type Output = Value;
      fn $op(self, rhs: $ty) -> Self::Output {
        self.$op(Value::new(rhs as f64, None))
      }
    }

    impl $Op<$ty> for &Value {
      type Output = Value;
      fn $op(self, rhs: $ty) -> Self::Output {
        self.clone().$op(rhs)
      }
    }

    impl $Op<Value> for $ty {
      type Output = Value;
      fn $op(self, rhs: Value) -> Self::Output {
        Value::new(self as f64, None).$op(rhs)
      }
    }

    impl $Op<&Value> for $ty {
      type Output = Value;
      fn $op(self, rhs: &Value) -> Self::Output {
        self.$op(rhs.clone())
      }
    }

    impl $OpAssign<$ty> for Value {
      fn $op_assign(&mut self, rhs: $ty) {
        let result = self.clone().$op(rhs);
        let result = result.inner();
        self.set_inner(result);
      }
    }
  };
}

macro_rules! impl_num {
  ($ty:ty) => {
    impl From<$ty> for Value {
      fn from(value: $ty) -> Self {
        Value::new(value as f64, None)
      }
    }
  };
}

macro_rules! impl_op {
  ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident) => {
    impl_op_ref!($Op, $op, $OpAssign, $op_assign);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, f64);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, f32);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, i8);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, u8);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, i16);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, u16);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, i32);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, u32);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, i64);
    impl_op_num!($Op, $op, $OpAssign, $op_assign, u64);

    impl $OpAssign<Value> for Value {
      fn $op_assign(&mut self, rhs: Value) {
        let result = self.clone().$op(rhs);
        let result = result.inner();
        self.set_inner(result);
      }
    }
  };
}

impl Add<Value> for Value {
  type Output = Value;

  fn add(self, rhs: Value) -> Self::Output {
    let a = self.data();
    let b = rhs.data();
    let mut result = Value::new(a + b, None);
    result.set_op(Some("+"));
    result.add_prev(&[&self, &rhs]);

    let mut tmp_rhs = rhs.clone();
    let mut tmp_self = self.clone();

    result.set_grad_fn(Box::new(move |grad| {
      tmp_self.add_grad(grad);
      tmp_rhs.add_grad(grad);
    }));

    result
  }
}

impl Sub<Value> for Value {
  type Output = Value;

  fn sub(self, rhs: Value) -> Self::Output {
    let a = self.data();
    let b = rhs.data();
    let mut result = Value::new(a - b, None);
    result.set_op(Some("-"));
    result.add_prev(&[&self, &rhs]);

    let mut tmp_self = self.clone();
    let mut tmp_rhs = rhs.clone();

    result.set_grad_fn(Box::new(move |grad| {
      tmp_self.add_grad(grad);
      tmp_rhs.add_grad(-grad);
    }));

    result
  }
}

impl Neg for Value {
  type Output = Value;

  fn neg(self) -> Self::Output {
    let mut result = Value::new(-self.data(), None);
    result.set_op(Some("-"));
    result.add_prev(&[&self]);

    let mut tmp_self = self.clone();

    result.set_grad_fn(Box::new(move |grad| {
      tmp_self.add_grad(-grad);
    }));

    result
  }
}

impl Neg for &Value {
  type Output = Value;

  fn neg(self) -> Self::Output {
    self.clone().neg()
  }
}

impl Mul<Value> for Value {
  type Output = Value;

  fn mul(self, rhs: Value) -> Self::Output {
    let a = self.data();
    let b = rhs.data();
    let mut result = Value::new(a * b, None);
    result.set_op(Some("*"));
    result.add_prev(&[&self, &rhs]);

    let mut tmp_self = self.clone();
    let mut tmp_rhs = rhs.clone();

    result.set_grad_fn(Box::new(move |grad| {
      tmp_self.add_grad(grad * b);
      tmp_rhs.add_grad(grad * a);
    }));

    result
  }
}

impl Div<Value> for Value {
  type Output = Value;

  fn div(self, rhs: Value) -> Self::Output {
    let a = self.data();
    let b = rhs.data();
    let mut result = Value::new(a / b, None);
    result.set_op(Some("/"));
    result.add_prev(&[&self, &rhs]);

    let mut tmp_rhs = rhs.clone();
    let mut tmp_self = self.clone();

    #[allow(clippy::suspicious_arithmetic_impl)]
    result.set_grad_fn(Box::new(move |grad| {
      tmp_self.add_grad(grad / b);
      tmp_rhs.add_grad(-grad * a / (b * b));
    }));

    result
  }
}

impl_num!(f64);
impl_num!(f32);
impl_num!(i8);
impl_num!(u8);
impl_num!(i16);
impl_num!(u16);
impl_num!(i32);
impl_num!(i64);
impl_num!(u32);
impl_num!(u64);

impl_op!(Add, add, AddAssign, add_assign);
impl_op!(Sub, sub, SubAssign, sub_assign);
impl_op!(Mul, mul, MulAssign, mul_assign);
impl_op!(Div, div, DivAssign, div_assign);

#[cfg(test)]
mod tests {
  use crate::Value;

  #[test]
  fn test_op_ref() {
    let x = Value::new(123.456, None);
    let y = Value::new(456.789, None);
    let expected = 580.245;

    {
      let z: Value = x.clone() + y.clone();
      assert_eq!(expected, z.data());
    }

    {
      let z: Value = x.clone() + &y;
      assert_eq!(expected, z.data());
    }

    {
      let z: Value = &x + y.clone();
      assert_eq!(expected, z.data());
    }

    {
      let z: Value = &x + &y;
      assert_eq!(expected, z.data());
    }
  }

  #[test]
  fn test_op_add() {
    let x: Value = 123.456.into();
    let y: Value = 456.789.into();
    let mut result = &x + &y;
    result.backward();

    assert_eq!(result.data(), 580.245);
    assert_eq!(result.grad(), 1.0);
    assert_eq!(x.grad(), 1.0);
    assert_eq!(y.grad(), 1.0);
  }

  #[test]
  fn test_op_sub() {
    let x: Value = 123.456.into();
    let y: Value = 456.789.into();

    let mut result = &x - &y;

    result.backward();

    assert_eq!(result.data(), -333.33299999999997);

    assert_eq!(result.grad(), 1.0);
    assert_eq!(x.grad(), 1.0);
    assert_eq!(x.grad(), 1.0);
    assert_eq!(y.grad(), -1.0);
  }

  #[test]
  fn test_op_neg() {
    let x: Value = 123.456.into();
    let mut result = -&x;

    result.backward();

    assert_eq!(result.data(), -123.456);
    assert_eq!(result.grad(), 1.0);
    assert_eq!(x.grad(), -1.0);
  }

  #[test]
  fn test_op_mul() {
    let x: Value = 123.456.into();
    let y: Value = 456.789.into();

    let mut result = &x * &y;

    result.backward();

    assert_eq!(result.data(), 56393.342784);
    assert_eq!(result.grad(), 1.0);
    assert_eq!(x.grad(), 456.789);
    assert_eq!(y.grad(), 123.456);
  }

  #[test]
  fn test_op_div() {
    let x: Value = 123.456.into();
    let y: Value = 456.789.into();

    let mut result = &x / &y;

    result.backward();

    assert_eq!(result.data(), 0.270269205256694);
    assert_eq!(result.grad(), 1.0);
    assert_eq!(x.grad(), 0.0021891945734244913);
    assert_eq!(y.grad(), -0.0005916718775117046);
  }
}
