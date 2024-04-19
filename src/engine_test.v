module microgradv

fn test_engine() {
	a := value(-4.0)
	assert a.data == -4.0
	assert a.grad == 0.0
	assert a.op == ''
	b := value(2.0)
	assert b.parents == []
	mut c := a.add(b)
	assert c.data == -2.0
	assert c.op == '+'
	assert c.parents == [a, b]
	mut d := a.mul(b).add(b.pow(3))
	assert d.data == 0
	assert d.op == '+'
	c = c.add(c.add(value(1)))
	c = c.add(value(1)).add(c).sub(a)
	assert c.data == -1
	d_1 := d.add(d.mul(value(2))).add(b.add(a))
	d = d_1.tanh()
	assert d.data == -0.9640275800758169
	assert d.op == 'tanh'
	assert d.parents == [d_1]
	d = d.add(value(3).mul(d)).add(b.sub(a)).relu()
	assert d.op == 'ReLu'
	assert d.data == 2.1438896796967324
	e := c.sub(d)
	assert e.parents == [c, d]
	assert e.op == '-'
	assert e.data == -3.1438896796967324
	assert e.grad == 0.0
	f := e.pow(2)
	assert f.op == '**'
	assert f.data == 9.884042318103623
	assert f.grad == 0.0
	assert e in f.parents
	mut g := f.div(value(2))
	assert f in g.parents
	assert g.op == '/'
	g = g.add(value(10).div(f))
	g = g.mul(value(2))
	assert g.op == '*'
	assert g.data == 11.90750593312356
	assert g.grad == 0.0
	assert g.data == 11.90750593312356
	g.backward()
	assert g.grad == 1.0
	assert a.grad == -10.10998355157139
	assert b.grad == 20.32762219152632
	assert e.grad == -5.000543596581886
	assert f.grad == 0.7952797499345223
}
