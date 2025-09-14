# OctonionLib

Перша **чиста Python бібліотека для октоніонів**.

## Можливості
- Клас `Octonion(a0..a7)`
- Арифметика: `+ - *`
- Спряження `conj()`
- Норма `norm()`
- Обернений елемент `inv()`
- Тести, що показують неасоціативність

## Приклад
```python
from octonionlib import Octonion

e1 = Octonion.basis(1)
e2 = Octonion.basis(2)
print(e1*e2)   # очікується e3
```
