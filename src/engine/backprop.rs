use std::collections::HashSet;

use crate::Value;

impl Value {
  #[allow(clippy::mutable_key_type)]
  pub fn backward(&mut self) {
    let mut topo = vec![];
    let mut visited = HashSet::new();

    build_topo(self, &mut topo, &mut visited);
    topo.reverse();

    self.set_grad(1.0);
    for mut v in topo {
      v.invoke_grad_fn();
    }
  }
}

#[allow(clippy::mutable_key_type)]
fn build_topo(
  value: &Value,
  topo: &mut Vec<Value>,
  visited: &mut HashSet<Value>,
) {
  if visited.contains(value) {
    return;
  }

  visited.insert(value.clone());
  for parent in value.prev() {
    build_topo(&parent, topo, visited);
  }

  topo.push(value.clone());
}
