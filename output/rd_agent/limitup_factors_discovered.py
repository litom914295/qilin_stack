
# 自动挖掘的涨停板一进二因子

def calculate_limitup_factors(data):
    '''
    计算涨停板预测因子
    
    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    
    Returns:
    --------
    pd.DataFrame: 因子数据
    '''
    result = data.copy()
    
    # 选中的因子
    selected_factors = ['seal_strength', 'limitup_time_score', 'leader_score']
    
    return result[selected_factors]

# 使用示例
# factors = calculate_limitup_factors(data)
