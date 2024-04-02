module microgradv

import math

pub struct Value {
pub:
        data f64 @[required]
pub mut:
        parents []&Value // default empty
        op      string   // default empty
        grad    f64 = 0.0 // default 0.0
}

pub fn (a &Value) add(b &Value) Value {
        mut out := Value{
                data: a.data + b.data
                parents: [a, b]
                op: '+'
        }
        return out
}

pub fn (a &Value) mul(b &Value) Value {
        mut out := Value{
                data: a.data * b.data
                parents: [a, b]
                op: '*'
        }
        return out
}

pub fn (a &Value) pow(b int) Value {
        mut out := Value{
                data: math.pow(a.data, b)
                parents: [&Value(a)]
                op: '**'
        }
        return out
}

fn (mut a Value) p_backward() {
        // Make sure the struct have 2 parents
        if a.op == '+' {
                a.parents[0].grad += a.grad
                a.parents[1].grad += a.grad
        }
}

pub fn (mut a Value) backward() {
		
		// build topologocal order
		// parents will become childrens.
		mut childrens := []&Value{}
		mut visited := []&Value{}
		build_topo(a, mut childrens, mut visited)
		a.grad = 1
		for mut child in childrens {
			child.p_backward()
		}
}

fn build_topo(a &Value, mut childrens []&Value, mut visited []&Value) {
		if a !in visited {
			visited.insert(0, a)
			for parent in a.parents {
				build_topo(parent, mut childrens, mut visited)
			}
			// by inserting at the first position, at the end we will have the parents of the value which backward was called on at the beginingg
			childrens.insert(0, a)
		} 
}
pub fn value(a f64) &Value {
	return &Value{
		data: a
	}
}
