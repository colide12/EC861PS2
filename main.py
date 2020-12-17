# This file estimates parameters for EC861 PS2.
#
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tabulate import tabulate
from multiprocessing import Pool, Process
from scipy.stats import norm
from scipy import optimize


class DataCleaner:
    def __init__(self, df, **kargs):
        self.df = df
        # drop values
        drop_dictionary = kargs['drop']
        for key in drop_dictionary.keys():
            self.df = self.df[self.df[key] != drop_dictionary[key]]

        # Define a container for market data
        self.df_market_level = pd.DataFrame([])

    def column_name(self):
        return self.df.column()

    def get_period_from_year_quarter(self, year_column='year', quarter_column='quarter', starting_year=1993):
        self.df['period'] = (self.df[year_column] - starting_year) * 4 + self.df[quarter_column]

    def get_city_pair(self, origin='origin', destination='dest', city_pair='city_pair'):
        self.df[city_pair] = self.df[origin] + self.df[destination]

    def get_2period_forward(
            self, column_name='active', new_column_name='active_next_period', period='period',
            on=('origin', 'dest', 'unique_carrier')
    ):
        if period not in self.df.columns:
            self.get_period_from_year_quarter()
        df2 = self.df[list(on + (period, column_name))]
        df_tmp = df2[period]
        df2[period] = df_tmp - 2  # forward 2 periods
        df2.rename(columns={column_name: new_column_name}, inplace=True)
        self.df = pd.merge(self.df, df2, on=list(on).append(period), validate='1:1')

    def get_firm_number_city2_market_share(
            self, market_column=('city_pair', 'period'), active='active', active_next_period='active_next_period',
            total_firm='total_firm', active_firm='active_firm', entries='entries', exits='exits',
            city_share='city_share', q='q', Q='Q',
            city2='city2',
            dpres=['dest_depart_sched', 'origin_depart_sched'], opres=['DAL_dds', 'DAL_odp']
    ):
        market_column = list(market_column)
        # get total number of firms in each market
        number_of_firms = self.df.groupby(by=market_column, as_index=False).size()
        number_of_firms.rename(columns={'size': total_firm}, inplace=True)
        self.df = pd.merge(self.df, number_of_firms, on=market_column, validate='m:1')

        # get total number of active firms in each market
        number_of_active_firms = self.df[self.df[active] == 1].groupby(by=market_column, as_index=False).size()
        number_of_active_firms.rename(columns={'size': active_firm}, inplace=True)
        self.df = pd.merge(self.df, number_of_active_firms, on=market_column, validate='m:1')

        # get total number of entries
        number_of_entries = self.df[self.df[active_next_period] == 1].groupby(by=market_column, as_index=False).size()
        number_of_entries.rename(columns={'size': entries}, inplace=True)
        self.df = pd.merge(self.df, number_of_entries, on=market_column, validate='m:1')

        # get total number of exits
        number_of_exits = self.df[self.df[active_next_period] == 0].groupby(by=market_column, as_index=False).size()
        number_of_exits.rename(columns={'size': exits}, inplace=True)
        self.df = pd.merge(self.df, number_of_exits, on=market_column, validate='m:1')

        # get market share
        self.df[city_share] = self.df[q] / self.df[Q]
        self.df[city_share] = self.df[city_share].fillna(0)
        # get city2
        destination_presence = ((self.df[dpres[0]] > 0) | (self.df[dpres[1]] > 0))
        origin_presence = ((self.df[opres[0]] > 0) | (self.df[opres[1]] > 0))

        self.df[city2] = (destination_presence & origin_presence).astype('int')

    def get_dist_sq(self, dist_sq='distance_squared', distance='distance'):
        self.df[dist_sq] = self.df[distance] * self.df[distance]

    def add_market_level_data(
            self, market_column=['city_pair', 'period'],
            market_variable=['total_firm', 'active_firm', 'entries', 'exits', 'distance', 'distance_squared',
                             'geo_mean_pop']
    ):
        # setup df_market_level
        self.df_market_level = self.df[market_column].drop_duplicates(ignore_index=True)
        self.df_market_level['market_index'] = self.df_market_level.index.values

        # add market_index to firm level data
        self.df = self.df.merge(self.df_market_level, on=market_column, validate='m:1')

        # add market level data to df_market_level
        df_tmp = self.df[market_column + market_variable].groupby(by=market_column, as_index=False).mean()
        self.df_market_level = pd.merge(self.df_market_level, df_tmp, on=market_column, validate='1:1')
        self.df_market_level.fillna(0)

    def scale_pop_dist(self, population='geo_mean_pop', distance='distance'):
        self.df[population] = self.df[population] / 10000000
        self.df[distance] = self.df[distance] / 1000


def nothing(arg1: int, arg2: str) -> bool:
    """함수의 문서화 문자열.

        Args:
            arg1 (int): 사실 함수에 이미 매개변수 형태가 있다면
                굳이 괄호로 표시해 줄 필요는 없습니다.
            arg2: 들여쓰기와 콜론(`:`)이 매개변수와 설명을
                구분합니다.

        Returns:
            bool: 이 경우에도 형태가 소스코드에 이미 있다면
                굳이 반복해서 쓸 필요는 없습니다.

        Raises:
            AttributeError: 예외 설명이 필요한 경우.

        Yields:
            출력값이 무언가를 나열하는 경우.

        Note:
            함께 알아두어야 할 사항이 있는 경우.

        `Args`나 `Returns` 등 각 섹션 사이에는 빈 줄이 하나 필요합니다.
        """
    return False


def berry_table_1(df: pd.DataFrame) -> str:
    """
        param:
            df: pd Dataframe which contains the number of total firms, active firms, entries and exits
            in each city_pair and period market
        return:
            table: LATEX style table in str
        Note:
            Table 1 of Berry (1992)
            THE JOINT FREQUENCY DISTRIBUTION OF ENTRY AND EXIT, IN PERCENT OF TOTAL MARKETS SERVED

    """

    # generate a table
    num_market = df.shape[0]
    number_of_rows = 4
    number_of_columns = 5
    container = np.ones((number_of_rows, number_of_columns))
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            if (i == 3) & (j == 3):
                container[i][j] = df[(df['entries'] >= i) & (df['exits'] >= j)].shape[0] / num_market
            elif (i == 3) & (j < 3):
                container[i][j] = df[(df['entries'] >= i) & (df['exits'] == j)].shape[0] / num_market
            elif (i < 3) & (j == 3):
                container[i][j] = df[(df['entries'] == i) & (df['exits'] >= j)].shape[0] / num_market
            else:
                container[i][j] = df[(df['entries'] == i) & (df['exits'] == j)].shape[0] / num_market
    container = container * 100

    # convert container(np.array) to table(list), and add row names
    table = []
    for i in range(number_of_rows):
        table.append(list(container[i]))
        if (i <= 2):
            table[i].insert(0, '{}'.format(i))
        elif (i == 3):
            table[i].insert(0, '3+')

    # header list
    headers = ['', '0', '1', '2', '3+']

    return tabulate(table, headers, tablefmt="latex", numalign="right", floatfmt=".2f")


def berry_table_2(df: pd.DataFrame, number_of_markets: int) -> str:
    """
        param:
            df: pd Dataframe which contains the number of total firms, active firms, entries and exits
            in each city_pair and period market
        return:
            table: LATEX style table in str
        Note:
            Table 2 of Berry (1992)
             NUMBER AND PERCENTAGE OF MARKETS ENTERED AND EXITED IN THE LARGE CITY SAMPLE, BY AIRLINE
    """

    # table to be reported
    df_markets_served = pd.DataFrame([])
    df_markets_potential = pd.DataFrame([])

    # get number of markets served by each carrier
    df_tmp = df[df['active']==1].groupby(by='unique_carrier', as_index=False).count().sort_values(
        by=['dest'], axis='rows', ascending=False, ignore_index=True
    )
    df_markets_served[['Airline', 'Number of Markets Served']] = df_tmp[['unique_carrier', 'dest']]
    df_markets_served['Percent of Markets Served'] = df_markets_served['Number of Markets Served']/number_of_markets
    df_tmp = df[df['active']==0].groupby(by='unique_carrier', as_index=False).count().sort_values(
        by=['dest'], axis='rows', ascending=False, ignore_index=True
    )
    df_markets_potential[['Airline', 'Number of Markets Potentially Entered']] = df_tmp[['unique_carrier', 'dest']]
    df_markets_potential['Percent of Markets Potentially Entered'] = df_markets_potential['Number of Markets Potentially Entered'] / number_of_markets

    # Merge two DataFrame
    df_markets_served = df_markets_served.merge(df_markets_potential, on=['Airline'], how='left').fillna(0)

    # set header
    headers = df_markets_served.columns

    return tabulate(df_markets_served, headers, tablefmt="latex", numalign="right", floatfmt=".2f")


def berry_table_4(df: pd.DataFrame) -> str:
    y = df[['active_firm']]
    x = df[['geo_mean_pop', 'distance', 'distance_squared', 'total_firm']]
    x_less_total_firm = df[['geo_mean_pop', 'distance', 'distance_squared']]

    x = sm.add_constant(x)
    x_less_total_firm = sm.add_constant(x_less_total_firm)

    number_of_variables = x.shape[1]

    # run OLS with full variables
    model = sm.OLS(y, x)
    results = model.fit(cov_type='HC0')

    # run OLS without total firm, city2
    model = sm.OLS(y, x_less_total_firm)
    results_less_total_firm = model.fit(cov_type='HC0')

    xmean = x.mean()
    xstd = x.std()
    # generate a container for a table
    table = []
    for i in range(number_of_variables):
        if results.params.index.values[i] == 'total_firm':
            table.append(
                [
                    '{}'.format(x.columns.values[i]),
                    '{:.2f}\n({:.2f})'.format(results.params.values[i], results.HC0_se[i]),
                    '--\n--',
                    '{:.2f}\n({:.2f})'.format(xmean[i], xstd[i])
                ]
            )
        elif results.params.index.values[i] == 'const':
            table.append(
                [
                    '{}'.format(x.columns.values[i]),
                    '{:.2f}\n({:.2f})'.format(results.params.values[i], results.HC0_se[i]),
                    '{:.2f}\n({:.2f})'.format(results_less_total_firm.params.values[i], results_less_total_firm.HC0_se[i]),
                    '--\n--',
                ]
            )
        else:
            table.append(
                [
                    '{}'.format(x.columns.values[i]),
                    '{:.2f}\n({:.2f})'.format(results.params.values[i], results.HC0_se[i]),
                    '{:.2f}\n({:.2f})'.format(results_less_total_firm.params.values[i], results_less_total_firm.HC0_se[i]),
                    '{:.2f}\n({:.2f})'.format(xmean[i], xstd[i])
                ]
            )

    # set header
    headers = ['Variable', '(1) OLS\nParameters\n(Std. Error)', '(2) OLS\nParameters\n(Std. Error)', 'Mean Value\nof Variable\n(Std. Dev.)']
    return tabulate(table, headers, tablefmt="latex", numalign="right", floatfmt=".2f")


def berry_table_5(df: pd.DataFrame) -> str:

    y = df['active_next_period']
    x = df[['geo_mean_pop', 'distance', 'distance_squared', 'city2']]
    x = sm.add_constant(x)
    number_of_variables = x.shape[1]

    x_less_city2 = df[['geo_mean_pop', 'distance', 'distance_squared']]
    x_less_city2 = sm.add_constant(x_less_city2)

    # run probit with the full set of variables
    probit_mod = sm.Probit(y, x)
    probit_res = probit_mod.fit()

    # run probit without city2
    probit_mod = sm.Probit(y, x_less_city2)
    probit_res_less_city2 = probit_mod.fit()

    # generate a container for a table
    table = []
    for i in range(number_of_variables):
        if probit_res.params.index.values[i] == 'city2':
            table.append(
                [
                    '{}'.format(x.columns.values[i]),
                    '{:.2f}\n({:.2f})'.format(probit_res.params.values[i], probit_res.bse.values[i]),
                    '--\n--'
                ]
            )
        else:
            table.append(
                [
                    '{}'.format(x.columns.values[i]),
                    '{:.2f}\n({:.2f})'.format(probit_res.params.values[i], probit_res.bse.values[i]),
                    '{:.2f}\n({:.2f})'.format(probit_res_less_city2.params.values[i],
                                              probit_res_less_city2.bse.values[i])
                ]
            )

    # set header
    headers = ['Variable', '(1) Probit\nParameters\n(Std. Error)', '(2) Probit\nParameters\n(Std. Error)']

    return tabulate(table, headers, tablefmt="latex", numalign="right", floatfmt=".2f")


def berry_table_6(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    # Generate the first column of table 6
    def objective_table_6_no_hetero(param, *args):
        y = args[0]
        x = args[1]
        x_plus_1 = args[2]
        pr_n0 = norm.cdf(x.dot(param))
        pr_n1 = norm.cdf(x_plus_1.dot(param))

        likelihood = pr_n0 - pr_n1
        log_likelihood = np.sum(np.log(likelihood))
        return -log_likelihood

    y = df2['active_next_period']
    x = df2[['geo_mean_pop', 'distance', 'distance_squared', 'city2', 'total_firm']]
    x = sm.add_constant(x)
    x_plus_1 = x
    x_plus_1['total_firm'] = x_plus_1['total_firm'] + 1
    x['total_firm'] = np.log(x['total_firm'])
    x_plus_1['total_firm'] = np.log(x_plus_1['total_firm'])
    a = objective_table_6_no_hetero([0] * 6, y, x, x_plus_1)
    # Generate the second column of table 6

    # Generate the third column of table 6

    return 'Hi'


def parallelize_dataframe(bs_data_list: np.array, func, n_cores: int = 4) -> np.array:
    """
        param:
            df:
            func: function to be applied
            n_cores: number of cores to use
        return:
            table: LATEX style table in str
        Note:
            THE JOINT FREQUENCY DISTRIBUTION OF ENTRY AND EXIT, IN PERCENT OF TOTAL MARKETS SERVED
    """

    bs_data_list_split = np.array_split(bs_data_list, n_cores)
    pool = Pool(n_cores)
    a = pool.map(func, bs_data_list_split)
    result_array = np.array()
    pool.close()
    pool.join()
    return result_array


def fmin_for_pool(args):
    arg = tuple(args[0])
    number_of_param = 10 + 1
    param0 = np.zeros((number_of_param, 1))
    minimum_no_order = optimize.fmin(
        objective_table_7, param0, args=arg, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None,
        full_output=True, disp=True, retall=False, callback=None, initial_simplex=None
    )

    return minimum_no_order


def generate_args_for_fmin(df_firm, df_market, num_simulation, num_firm, num_market, market_index, is_no_order):
    # simulate errors
    seed = 2375
    uik = norm.rvs(size=(num_firm, num_simulation), random_state=seed)
    ui0 = norm.rvs(size=(num_market, num_simulation), random_state=seed)

    # attach simulated errors to market_index

    simulated_errors_firm = pd.concat([df_firm[market_index], pd.DataFrame(uik)], axis=1)
    simulated_errors_market = pd.concat([df_market[market_index], pd.DataFrame(ui0)], axis=1)
    simulated_errors_market = pd.merge(df_firm[market_index], simulated_errors_market, on=market_index)

    # observed data
    y = df_firm['active_next_period']
    x = df_firm[['geo_mean_pop', 'distance', 'distance_squared', 'total_firm', 'city2', 'origin_pass', 'dest_pass',
                 'DAL_opass', 'DAL_dpass']]
    x = sm.add_constant(x)
    x['total_firm'] = np.log(x['total_firm'])
    z = df_market[['geo_mean_pop', 'distance', 'distance_squared', 'total_firm', 'entries']]
    z = np.array(sm.add_constant(z))
    number_of_param = 10 + 1

    # generate matrix of total number of firms
    number_of_entries = np.zeros(shape=(x.shape[0], num_simulation))
    for i in range(num_simulation):
        number_of_entries[:, i] = df_firm['entries']

    # generate container for data-error DataFrame
    data_container_x = np.zeros(shape=(num_simulation, x.shape[0], number_of_param + 1))
    data_container_y = np.zeros(shape=(num_simulation, x.shape[0], 1))
    # generate data-error DataFrame
    for i in range(num_simulation):
        data_container_x[i, :, :] = np.array(
            pd.concat([x, simulated_errors_market[i], simulated_errors_firm[i]], axis=1)
        )
        data_container_y[i, :, 0] = np.array(y)

    args = (data_container_x, data_container_y, df_firm[market_index], number_of_entries, z, is_no_order)
    return args


def objective_table_7(param, *args):
    data_container_x = args[0]
    data_container_y = args[1]
    df_firm_market_index = args[2]
    number_of_active_firm = args[3]
    data_market = args[4]
    location_of_number_of_entries = data_market.shape[1] - 2
    is_no_order = args[5]

    number_of_explanatory_variables = len(param) - 1  # param contains rho
    param = np.concatenate((param, np.array([np.sqrt(1 - param[-1])])), axis=0)

    profit_simulated = np.dot(data_container_x, param)
    profit_simulated = np.reshape(profit_simulated, (profit_simulated.shape[1], profit_simulated.shape[0]))
    profit_simulated = pd.concat([df_firm_market_index, pd.DataFrame(profit_simulated)], axis=1)

    if not is_no_order:
        # calculate vij
        df_rank = profit_simulated.groupby(by=list(df_firm_market_index.columns.values), as_index=False).rank(
            ascending=False)
        vij = np.mean((number_of_active_firm - np.array(df_rank) >= 0), axis=1)

    # calculate vi0

    df_tmp = profit_simulated.groupby(by=list(df_firm_market_index.columns.values)).sum()
    number_of_market = df_tmp.shape[0]

    df_tmp = df_tmp.mean(axis=1)
    vi0 = np.array(df_tmp) - np.array(data_market[:, location_of_number_of_entries - 1])

    # generate moment conditions
    moment_squared = 0
    x = np.squeeze(data_container_x[0, :, 0:number_of_explanatory_variables])
    if not is_no_order:
        for i in range(number_of_explanatory_variables):
            moment = np.multiply(x[:, i], vij)
            moment_squared += np.dot(moment, moment)

    for i in range(location_of_number_of_entries - 1):
        moment = np.multiply(data_market[:, i], vi0)
        moment_squared += np.dot(moment, moment)

    moment_squared /= number_of_market
    print(moment_squared)
    return moment_squared


def berry_table_7(df_firm: pd.DataFrame, df_market: pd.DataFrame):
    # def objective_table_7_no_order(param, *args):
    #     # firm level data
    #     data_container_x = args[0]
    #     # number of firms enter
    #     data_entry = args[1]
    #     # market numbers used
    #     df_market_index = args[2]
    #     # param = [..., rho, sqrt(1-rho^{2})]
    #     param = np.concatenate((param, np.array([np.sqrt(1-param[-1])])), axis=0)
    #     profit_simulated = np.dot(data_container_x, param)
    #     profit_simulated = np.reshape(profit_simulated, (profit_simulated.shape[1], profit_simulated.shape[0]))
    #     profit_simulated = (profit_simulated >= 0)
    #     df_tmp = pd.concat([df_market_index, pd.DataFrame(profit_simulated)], axis=1)
    #     df_tmp = df_tmp.groupby(by=list(df_market_index.columns.values)).sum()
    #     df_tmp = df_tmp.mean(axis=1)
    #     vi0 = np.array(df_tmp) - np.array(data_entry)
    #     a = np.dot(vi0, vi0)
    #
    #     return a

    num_simulation = 100
    num_firm = df_firm.shape[0]
    num_market = df_market.shape[0]
    number_of_bootstrap = 4
    market_index = ['market_index']
    is_no_order = True
    number_of_param = 10 + 1
    args = generate_args_for_fmin(df_firm, df_market, num_simulation, num_firm, num_market, market_index, is_no_order)

    # generate container for the bootstrap
    bs_firm_container = []
    bs_market_container = []
    bs_result_no_order_container = []
    bs_result_profitable_container = []
    # generate data for the bootstrap
    param0 = np.zeros((number_of_param, 1))

    for i in range(number_of_bootstrap):
        is_no_order = True
        # args = (data_container_x, data_container_y, df_firm[market_index], number_of_entries, z, is_no_order)

        randlist = pd.DataFrame(index=np.random.randint(num_market, size=num_market))
        bs_market_data = df_market[market_index].merge(randlist, left_index=True, right_index=True, how='right')
        bs_market_data['market_index'] = range(num_market)
        bs_firm_data = bs_market_data.merge(df_firm, on=market_index, validate='m:m')
        args_tmp = generate_args_for_fmin(df_firm, df_market, num_simulation, num_firm, num_market, market_index,
                                          is_no_order)
        bs_firm_container.append(args_tmp)

    # optimization
    # Generate the first column of table 7, without entry order
    param0 = np.zeros((number_of_param, 1))
    # minimum_no_order = optimize.fmin(
    #     objective_table_7, param0, args=args, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None,
    #     full_output=1, disp=1, retall=0, callback=None, initial_simplex=None
    # )
    # # Generate the second column of table 7, the most profitable moves first
    # is_no_order = False
    # args = (data_container_x, data_container_y, df_firm[market_index], number_of_entries, z, is_no_order)
    # minimum_profitability = optimize.fmin(
    #     objective_table_7, param0, args=args, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None,
    #     full_output=1, disp=1, retall=0, callback=None, initial_simplex=None
    # )
    # run bootstrap for table 7, without entry order

    # result = parallelize_dataframe(bs_firm_container, fmin_for_pool)
    # for i in range(number_of_bootstrap):
    #     is_no_order = True
    #     args = (data_container_x, data_container_y, df_firm[market_index], number_of_entries, z, is_no_order)
    #     bs_result_no_order_container.append(
    #         optimize.fmin(
    #             objective_table_7, param0, args=args, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None,
    #             full_output=1, disp=1, retall=0, callback=None, initial_simplex=None
    #         )
    #     )
    return bs_firm_container


if __name__ == '__main__':
    # import data
    df_to_clean = pd.read_stata('IO2020.dta')

    # clean data
    cleaning = DataCleaner(df_to_clean, drop={'stop': 1, 'origin': 'DAL'})
    cleaning.scale_pop_dist()
    cleaning.get_period_from_year_quarter()
    cleaning.get_city_pair()
    cleaning.get_2period_forward()
    cleaning.get_firm_number_city2_market_share()
    cleaning.get_dist_sq()
    cleaning.add_market_level_data()

    # result of the cleaning
    df_cleaned = cleaning.df
    df_market_level = cleaning.df_market_level

    # generate table 1 and save
    with open("berry_table1.tex", "w") as f:
        table1_latex = berry_table_1(df_market_level)
        f.write(table1_latex)

    # generate table 2 and save
    with open("berry_table2.tex", "w") as f:
        table2_latex = berry_table_2(df_cleaned, df_market_level.shape[0])
        f.write(table2_latex)

    # table 3 is not generated

    # generate table 4 and save
    with open("berry_table4.tex", "w") as f:
        table4_latex = berry_table_4(df_market_level)
        f.write(table4_latex)

    berry_table_5(df_cleaned)
    bs_data_list = berry_table_7(df_cleaned, df_market_level)

    n_cores = 2
    bs_data_list_split = np.array_split(bs_data_list, n_cores)
    pool = Pool(n_cores)
    a = pool.map(fmin_for_pool, bs_data_list_split)
    print(a)
    pool.close()
    pool.join()
