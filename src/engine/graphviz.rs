use graphviz_rust::{
  cmd::Format,
  dot_generator::*,
  dot_structures::*,
  exec,
  printer::{DotPrinter, PrinterContext},
};
use linked_hash_set::LinkedHashSet;

use crate::Value;

impl Value {
  pub fn into_dot(&self) -> Graph {
    let (mut nodes, mut edges) = self.trace_graph();
    nodes.sort_by_key(|x| x.id());
    edges.sort_by_key(|a| a.0.id());

    let mut stmts = vec![stmt!(attr!("rankdir", "LR"))];

    for node in nodes.iter() {
      let node_id = format!("{}", node.id());
      let name = node.label();
      let label = format!(
        "\"{{ {} | data {:.4} | grad {:0.4} }}\"",
        name,
        node.data(),
        node.grad()
      );
      stmts.push(stmt!(node!(node_id;
        attr!("shape", "record"),
        attr!("label", label)
      )));

      if let Some(op) = node.op() {
        let op_id = format!("\"{}:op:{}\"", node.id(), op);
        let op_label = format!("\"{}\"", op);
        stmts.push(stmt!(node!(op_id;
          attr!("label", op_label)
        )));
        stmts.push(stmt!(edge!(
              node_id!(op_id) => node_id!(node_id)
        )));
      }
    }

    for (n1, n2) in edges.iter() {
      let n1_id = format!("{}", n1.id());
      let n2_id = format!("\"{}:op:{}\"", n2.id(), n2.op().unwrap_or_default());
      stmts.push(stmt!(edge!(
        node_id!(n1_id) => node_id!(n2_id)
      )));
    }

    Graph::DiGraph {
      id: id!("abc"),
      strict: true,
      stmts,
    }
  }

  pub fn into_dot_str(&self) -> String {
    let dot = self.into_dot();
    dot.print(&mut PrinterContext::default())
  }

  pub fn into_svg(&self) -> String {
    let dot = self.into_dot();
    let svg_data = exec(
      dot,
      &mut PrinterContext::default(),
      vec![Format::Svg.into()],
    )
    .unwrap();

    String::from_utf8(svg_data).unwrap()
  }

  fn trace_graph(&self) -> (Vec<Value>, Vec<(Value, Value)>) {
    let mut nodes = LinkedHashSet::<Value>::new();
    let mut edges = LinkedHashSet::<(Value, Value)>::new();

    fn build(
      node: &Value,
      nodes: &mut LinkedHashSet<Value>,
      edges: &mut LinkedHashSet<(Value, Value)>,
    ) {
      if nodes.contains(node) {
        return;
      }
      nodes.insert(node.clone());
      for child in node.prev().iter() {
        edges.insert((child.clone(), node.clone()));
        build(child, nodes, edges);
      }
    }

    build(self, &mut nodes, &mut edges);

    let nodes = nodes.iter().cloned().collect::<Vec<_>>();
    let edges = edges.iter().cloned().collect::<Vec<_>>();

    (nodes, edges)
  }
}

#[cfg(test)]
mod tests {
  use crate::Value;

  #[test]
  fn test_value_into_svg() {
    let x = Value::new(1.0, Some("x"));
    let y: Value = x * 2 + 1;
    let dot = y.into_dot_str();

    assert!(dot.contains("x"));

    // write to /tmp/test.svg
    let svg = y.into_svg();
    use std::fs::File;
    use std::io::Write;
    let mut file = File::create("/tmp/test.svg").unwrap();
    file.write_all(svg.as_bytes()).unwrap();
  }
}
