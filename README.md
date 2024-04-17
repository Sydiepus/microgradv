# microgradv

A tiny Autograd engine, about 250 lines of code in total!

An implementation of Andrej karpathy's [micrograd](https://github.com/karpathy/micrograd) in Vlang that I wrote as exercise to learn about neural networks and V. 

> [!NOTE]
> Due to V's limitations on operator overloading using this implementation is painful to the eyes.
> I hope this will be useful for educational purposes like the original implementation.

## Installation

```
v install Sydiepus.microgradv
```

## Example Usage

```v
import sydiepus.microgradv

a := microgradv.value(20)
b := microgradv.value(58)
c := a.add(b)
d := c.mul(a)
e := d.pow(2)
f := e.div(a)
g := f.sub(b)
mut h := g.relu()
h.backward()
println(a.grad) // prints 9204.0
println(b.grad) // prints 3119.0
println(c.grad) // prints 3120.0
println(d.grad) // prints 156.0
println(e.grad) // prints 0.05
println(f.grad) // prints 1.0
println(g.grad) // prints 1.0
println(h.grad) // prints 1.0
```

## Training a neural network

In progress
