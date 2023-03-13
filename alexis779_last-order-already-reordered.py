#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

class Instacart:
  def load_csv(self, file_name):
    return pd.read_csv("../input/" + file_name)

  def first_row(self, rows):
    return rows.iloc[0]

  def intersection_ratio(self, products1, products2):
    if len(products2) == 0:
      return 1
    else:
      return float(len(set(products1).intersection(set(products2)))) / len(products2)

  def predict_user(self, order_products):
    row = {}

    row["user_id"] = order_products.iloc[0].user_id

    orders = order_products.groupby("order_id", as_index=False)       .agg({"order_number" : self.first_row})       .sort_values("order_number")

    current_order_id = orders.tail(1).iloc[0].order_id
    reordered_current = order_products[(order_products.order_id == current_order_id) & (order_products.reordered == 1)].product_id

    if len(orders) >= 2:
      [last_order_id, current_order_id] = orders.tail(2).order_id

      last_order = order_products[order_products.order_id == last_order_id].product_id
      reordered_past = order_products[(order_products.order_id != current_order_id) & (order_products.reordered == 1)].product_id

      """ Estimate how reordered products in current order were present in last order """
      row["last_order_ratio"] = self.intersection_ratio(last_order, reordered_current)

      """ Estimate how reordered products in current order were already reordered in past orders """
      row["already_reordered_ratio"] = self.intersection_ratio(reordered_past, reordered_current)

    return pd.Series(row)

  def predict(self):
    pd.set_option("display.width", 1000)

    # get orders associated to users from the training set
    orders = self.load_csv("orders.csv")
    user_ids = orders[orders.eval_set == "train"]["user_id"]
    user_ids = pd.DataFrame(user_ids)
    order_columns = ["order_id", "user_id", "order_number"]
    orders = orders.merge(user_ids)[order_columns]

    # get order products associated to those orders
    order_products = pd.concat([self.load_csv("order_products__prior.csv"), self.load_csv("order_products__train.csv")])
    order_product_columns = ["order_id", "product_id", "reordered"]
    order_products = order_products[order_product_columns].merge(orders)

    # get aggregated users from those order products
    users = order_products.groupby("user_id").apply(self.predict_user)
    df = pd.DataFrame(users)
    return df[["last_order_ratio", "already_reordered_ratio"]].mean()

Instacart().predict()

