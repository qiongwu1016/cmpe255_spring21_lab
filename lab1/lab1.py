import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        # TODO:
        # Load data from data/chipotle.tsv file using Pandas library and
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')

    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())

    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        print(self.chipo.count())
        return len(self.chipo.index)

    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info(verbose=True))
        pass

    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        print(len(self.chipo.columns))
        return len(self.chipo.columns)

    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        my_list = list(self.chipo)
        print(my_list)
        pass

    def most_ordered_item(self):
        # TODO
        item_name = None
        order_id = -1
        quantity = -1
        popular_item = self.chipo.groupby('item_name').agg(
            {'order_id': "sum", 'quantity': "sum"}).sort_values(by='quantity', ascending=False)

        #print(popular_item.head(30))
        label = popular_item.index.values[0]
        order_id = popular_item.loc[label]['order_id']
        quantity = popular_item.loc[label]['quantity']
        print(label, order_id, quantity)
        return label, order_id, quantity

    def total_item_orders(self) -> int:
        # TODO How many items were orderd in total?
        item_orders = self.chipo.agg({'quantity': 'sum'})
        print(item_orders.get(key="quantity"))
        return item_orders.get(key="quantity")

    def total_sales(self) -> float:
        # TODO
        # 1. Create a lambda function to change all item prices to float.
        self.chipo['item_price'] = self.chipo['item_price'].apply(
            lambda x: float(x.replace('$', '')))
        #print(self.chipo.head(10))
        # 2. Calculate total sales.
        # print(self.chipo.info(verbose=True))
        self.chipo['sub_total'] = self.chipo['item_price'] * \
            self.chipo['quantity']
        sales = self.chipo.agg({'sub_total': 'sum'})
        print(sales.get(key='sub_total'))
        return sales.get(key='sub_total')

    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        num_orders = len(self.chipo.order_id.unique())
        print(num_orders)
        return num_orders

    def average_sales_amount_per_order(self) -> float:
        # TODO
        self.chipo['sub_total'] = self.chipo['item_price'] * \
            self.chipo['quantity']
        df = self.chipo.groupby('order_id').agg({'sub_total': 'sum'})
        average_amount = df['sub_total'].mean().round(2)
        print(average_amount)
        return average_amount

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        # print(self.chipo.head())
        items = self.chipo.item_name.unique()
        print(len(items))
        return len(items)

    def plot_histogram_top_x_popular_items(self, x: int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        print(letter_counter)
        # TODO
        # 1. convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict({k: [v] for k, v in letter_counter.items()}, orient='index', columns=['n'])
        print(df.head(10))

        # 2. sort the values from the top to the least value and slice the first 5 items
        df = df.sort_values(by=['n'], ascending = False).head(5)
        # 3. create a 'bar' plot from the DataFrame
        plt.figure()
        df.plot(kind="bar")
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
        plt.title('Most popular items')
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        pass

    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        df = self.chipo[['order_id', 'quantity', 'sub_total']].groupby(['order_id']).sum()
        print(df.head(10))
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        plt.figure()
        df.plot.scatter(x='sub_total', y='quantity', s=50, c='blue')
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        plt.title('Numer of items per order price')
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.show(block=True)
        pass


def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()


if __name__ == "__main__":
    # execute only if run as a script
    test()
