module microgradv

import math

pub struct Value {
mut:
	val_backward fn () = fn () {
		return
	}
pub:
	data f64    @[required]
	op   string // default empty
pub mut:
	parents []&Value // default empty
	grad    f64 = 0.0 // default 0.0
}

pub fn (a &Value) add(b &Value) &Value {
	mut out := &Value{
		data: a.data + b.data
		parents: [a, b]
		op: '+'
	}
	out.val_backward = fn [mut out] () {
		out.parents[0].grad += out.grad
		out.parents[1].grad += out.grad
	}
	return out
}

pub fn (a &Value) mul(b &Value) &Value {
	mut out := &Value{
		data: a.data * b.data
		parents: [a, b]
		op: '*'
	}
	out.val_backward = fn [mut out] () {
		out.parents[0].grad += out.parents[1].data * out.grad
		out.parents[1].grad += out.parents[0].data * out.grad
	}

	return out
}

pub fn (a &Value) pow(b f64) &Value {
	mut out := &Value{
		data: math.pow(a.data, b)
		parents: [a]
		op: '**'
	}
	out.val_backward = fn [mut out, b] () {
		out.parents[0].grad += (b * math.pow(out.parents[0].data, b - f64(1))) * out.grad
	}
	return out
}

pub fn (mut a Value) backward() {
	// build topological order
	// parents will become children.
	mut children := []&Value{}
	mut visited := []&Value{}
	build_topo(a, mut children, mut visited)
	a.grad = 1
	for child in children {
		child.val_backward()
	}
}

fn build_topo(a &Value, mut children []&Value, mut visited []&Value) {
	if a !in visited {
		visited.insert(0, a)
		for parent in a.parents {
			build_topo(parent, mut children, mut visited)
		}
		// by inserting at the first position, at the end we will have the parents of the value which backward was called on at the beginning
		children.insert(0, a)
	}
}

pub fn value(a f64) &Value {
	return &Value{
		data: a
	}
}
