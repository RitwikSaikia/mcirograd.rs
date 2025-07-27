use lazy_static::lazy_static;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU32;
use std::{cell::RefCell, rc::Rc};

mod backprop;
mod fns;
mod graphviz;
mod ops;

#[derive(Clone)]
pub struct Value {
  inner: Rc<RefCell<ValueInner>>,
}

pub(crate) struct ValueInner {
  id: u32,
  data: f64,
  grad: f64,
  label: String,
  op: Option<String>,
  prev: Vec<Value>,
  grad_fn: Option<Box<dyn FnMut(f64)>>,
}

impl Value {
  pub fn new(data: f64, label: Option<&str>) -> Self {
    lazy_static! {
      static ref ID_COUNTER: AtomicU32 = AtomicU32::new(0);
    }
    let inner = Rc::new(RefCell::new(ValueInner {
      id: ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
      data,
      grad: 0.0,
      label: label.unwrap_or_default().to_string(),
      op: None,
      prev: vec![],
      grad_fn: None,
    }));
    Self { inner }
  }

  pub(crate) fn inner(self) -> Rc<RefCell<ValueInner>> {
    self.inner.clone()
  }

  pub(crate) fn set_inner(&mut self, inner: Rc<RefCell<ValueInner>>) {
    self.inner = inner;
  }

  pub fn id(&self) -> u32 {
    self.inner.borrow().id
  }

  pub fn data(&self) -> f64 {
    self.inner.borrow().data
  }

  pub fn set_data(&mut self, data: f64) {
    self.inner.borrow_mut().data = data;
  }

  pub fn grad(&self) -> f64 {
    self.inner.borrow().grad
  }

  pub fn set_grad(&mut self, grad: f64) {
    self.inner.borrow_mut().grad = grad;
  }

  pub fn zero_grad(&mut self) {
    self.inner.borrow_mut().grad = 0.0;
  }

  pub fn add_grad(&mut self, grad: f64) {
    self.inner.borrow_mut().grad += grad;
  }

  pub fn label(&self) -> String {
    self.inner.borrow().label.to_string()
  }

  pub fn set_label(&mut self, label: &str) {
    self.inner.borrow_mut().label = label.to_string();
  }

  pub fn op(&self) -> Option<String> {
    self.inner.borrow().op.clone()
  }

  pub(crate) fn set_op(&self, op: Option<&str>) {
    self.inner.borrow_mut().op = op.map(|x| x.to_string());
  }

  pub fn prev(&self) -> Vec<Value> {
    self.inner.borrow().prev.clone()
  }

  pub(crate) fn add_prev(&mut self, prev: &[&Value]) {
    for &v in prev {
      self.inner.borrow_mut().prev.push(v.clone());
    }
  }

  pub fn invoke_grad_fn(&mut self) {
    let grad = self.grad();
    if let Some(grad_fn) = self.inner.borrow_mut().grad_fn.as_mut() {
      grad_fn(grad);
    }
  }

  pub fn set_grad_fn(&mut self, grad_fn: Box<dyn FnMut(f64)>) {
    self.inner.borrow_mut().grad_fn = Some(Box::new(grad_fn));
  }
}

impl fmt::Debug for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let inner = self.inner.borrow();
    let mut w = f.debug_struct("Value");
    w.field("data", &inner.data).field("grad", &inner.grad);

    if !inner.label.is_empty() {
      let label = inner.label.as_str();
      w.field("label", &label);
    }

    if let Some(op) = &inner.op {
      w.field("op", op);
    }

    w.finish()
  }
}

impl Eq for Value {}

impl PartialEq for Value {
  fn eq(&self, other: &Self) -> bool {
    Rc::ptr_eq(&self.inner, &other.inner)
  }
}

impl Eq for ValueInner {}

impl PartialEq for ValueInner {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl Hash for Value {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.inner.borrow().id.hash(state);
  }
}

impl Hash for ValueInner {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.id.hash(state);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_example_1() {
    let x = Value::new(-4.0, None);
    let z: Value = 2.0 * &x + 2.0 + &x;
    let q = z.relu() + &z * &x;
    let h = (&z * &z).relu();
    let mut y = &h + &q + &q * &x;

    y.backward();

    assert_eq!(-20.0, y.data());
    assert_eq!(46.0, x.grad());
  }

  #[test]
  fn test_debug() {
    let mut v = Value::new(123.456, Some("abcd"));
    v.set_grad(456.789);

    assert_eq!(
      "Value { data: 123.456, grad: 456.789, label: \"abcd\" }",
      format!("{:?}", v)
    );
  }
}
