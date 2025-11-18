from decimal import Decimal,ROUND_HALF_UP

# The original round function of Python has different defination, 
# like result of round(1.335,2) is 1.33 rather than 1.34,
# So we need use decimal to creat a right round function
def rightRound(num: float | str, keep_n: int=0) -> float:
    if isinstance(num,float):
        num=str(num)
    result = Decimal(num).quantize((Decimal('0.' + '0'*keep_n)),rounding=ROUND_HALF_UP)

    return float(result)