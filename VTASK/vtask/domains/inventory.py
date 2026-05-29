"""
Domain 3: Inventory & Logistics Reasoning
"""
from __future__ import annotations
import random
from vtask.base import TaskEntry, TaskGenerator

PRODUCTS = ["laptop", "chair", "monitor", "keyboard", "notebook", "pen", "stapler",
            "desk", "headset", "webcam", "tablet", "printer"]


class InventoryGenerator(TaskGenerator):
    domain = "inventory"
    difficulty_range = (1, 5)

    def _simulate_single(self, initial: int, demand_per_day: int, days: int,
                         reorder_point: int = 0, reorder_qty: int = 0,
                         lead_time: int = 0) -> dict:
        """Simulate a single-item inventory. Returns day-by-day state."""
        stock = initial
        orders_in_transit = []  # list of (arrive_day, qty)
        stockout_day = None
        stocks = [initial]

        for day in range(1, days + 1):
            # Receive incoming orders
            arrived = [qty for (arrive, qty) in orders_in_transit if arrive == day]
            for qty in arrived:
                stock += qty
            orders_in_transit = [(a, q) for (a, q) in orders_in_transit if a != day]

            # Consume demand
            stock = max(0, stock - demand_per_day)
            if stock == 0 and stockout_day is None and demand_per_day > 0:
                stockout_day = day

            # Reorder check
            if reorder_point > 0 and stock <= reorder_point and reorder_qty > 0:
                # Check if we already have an order in transit
                if not orders_in_transit:
                    orders_in_transit.append((day + lead_time, reorder_qty))

            stocks.append(stock)

        return {"stocks": stocks, "stockout_day": stockout_day, "final_stock": stocks[-1]}

    def _simulate(self, params: dict) -> int:
        mode = params.get("mode", "stockout_day")

        if mode == "stockout_day":
            initial = params["initial"]
            demand = params["demand_per_day"]
            if demand <= 0:
                return -1
            # Day when stock first hits 0
            day = (initial + demand - 1) // demand
            return day

        elif mode == "stock_on_day":
            result = self._simulate_single(
                initial=params["initial"],
                demand_per_day=params["demand_per_day"],
                days=params["query_day"],
                reorder_point=params.get("reorder_point", 0),
                reorder_qty=params.get("reorder_qty", 0),
                lead_time=params.get("lead_time", 0),
            )
            return result["stocks"][params["query_day"]]

        elif mode == "two_items":
            # Return stock of item1 on query_day
            r1 = self._simulate_single(
                initial=params["item1_initial"],
                demand_per_day=params["item1_demand"],
                days=params["query_day"],
                reorder_point=params.get("item1_reorder_point", 0),
                reorder_qty=params.get("item1_reorder_qty", 0),
                lead_time=params.get("lead_time", 0),
            )
            r2 = self._simulate_single(
                initial=params["item2_initial"],
                demand_per_day=params["item2_demand"],
                days=params["query_day"],
                reorder_point=params.get("item2_reorder_point", 0),
                reorder_qty=params.get("item2_reorder_qty", 0),
                lead_time=params.get("lead_time", 0),
            )
            return r1["stocks"][params["query_day"]] + r2["stocks"][params["query_day"]]

        elif mode == "seasonal":
            # Two demand rates, week 1 vs week 2+
            stock = params["initial"]
            demand_w1 = params["demand_week1"]
            demand_w2 = params["demand_week2"]
            query_day = params["query_day"]
            for day in range(1, query_day + 1):
                demand = demand_w1 if day <= 7 else demand_w2
                stock = max(0, stock - demand)
            return stock

        elif mode == "multi_echelon":
            # Central warehouse + 2 stores, transfers
            wh = params["warehouse_initial"]
            s1 = params["store1_initial"]
            s2 = params["store2_initial"]
            wh_demand = params["warehouse_demand"]
            s1_demand = params["store1_demand"]
            s2_demand = params["store2_demand"]
            transfer_day = params["transfer_day"]
            transfer_qty = params["transfer_qty"]
            query_day = params["query_day"]

            for day in range(1, query_day + 1):
                # Consume demand
                wh = max(0, wh - wh_demand)
                s1 = max(0, s1 - s1_demand)
                s2 = max(0, s2 - s2_demand)
                # Transfer from warehouse to stores on transfer_day
                if day == transfer_day:
                    actual = min(wh, transfer_qty)
                    wh -= actual
                    half = actual // 2
                    s1 += half
                    s2 += actual - half

            return wh + s1 + s2

        return 0

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)
        product = rng.choice(PRODUCTS)

        if difficulty == 1:
            initial = rng.randint(40, 200)
            demand = rng.randint(3, 15)
            stockout_day = (initial + demand - 1) // demand
            params = {"mode": "stockout_day", "initial": initial, "demand_per_day": demand}
            answer = str(stockout_day)
            question = (
                f"You start with {initial} units of {product} in stock. "
                f"You sell {demand} units per day. "
                f"On which day do you run out of stock? "
                f"(Day 1 is the first day of selling.)"
            )
            distractors = [str(stockout_day - 2), str(stockout_day + 1), str(stockout_day + 3)]
            distractors = [d for d in distractors if d != answer and int(d) > 0]
            metadata = {"params": params, "correct_answer": stockout_day, "mode": "stockout_day"}

        elif difficulty == 2:
            initial = rng.randint(30, 100)
            demand = rng.randint(3, 8)
            reorder_point = rng.randint(10, 25)
            reorder_qty = rng.randint(30, 80)
            lead_time = rng.randint(2, 5)
            query_day = rng.randint(10, 20)
            params = {
                "mode": "stock_on_day",
                "initial": initial,
                "demand_per_day": demand,
                "reorder_point": reorder_point,
                "reorder_qty": reorder_qty,
                "lead_time": lead_time,
                "query_day": query_day,
            }
            correct = self._simulate(params)
            answer = str(correct)
            question = (
                f"You start with {initial} units of {product}. "
                f"You sell {demand} units per day. "
                f"When stock hits {reorder_point} or below, you place a reorder of {reorder_qty} units. "
                f"The reorder takes {lead_time} days to arrive. "
                f"How many units do you have at the end of day {query_day}?"
            )
            distractors = [str(correct - 3), str(correct + 5), str(correct + 10)]
            distractors = [d for d in distractors if d != answer and int(d) >= 0]
            metadata = {"params": params, "correct_answer": correct, "mode": "stock_on_day"}

        elif difficulty == 3:
            item1 = rng.choice(PRODUCTS)
            item2 = rng.choice([p for p in PRODUCTS if p != item1])
            i1_init = rng.randint(50, 120)
            i2_init = rng.randint(50, 120)
            i1_demand = rng.randint(3, 10)
            i2_demand = rng.randint(3, 10)
            cap = rng.randint(150, 250)
            reorder_pt = rng.randint(15, 30)
            reorder_qty = rng.randint(30, 60)
            lead_time = rng.randint(2, 4)
            query_day = rng.randint(8, 15)
            params = {
                "mode": "two_items",
                "item1_initial": i1_init,
                "item1_demand": i1_demand,
                "item2_initial": i2_init,
                "item2_demand": i2_demand,
                "item1_reorder_point": reorder_pt,
                "item1_reorder_qty": reorder_qty,
                "item2_reorder_point": reorder_pt,
                "item2_reorder_qty": reorder_qty,
                "lead_time": lead_time,
                "query_day": query_day,
            }
            correct = self._simulate(params)
            answer = str(correct)
            question = (
                f"A warehouse (capacity {cap} units total) tracks two items: {item1} and {item2}.\n"
                f"  {item1}: starts with {i1_init} units, sells {i1_demand}/day, "
                f"reorders {reorder_qty} units when stock ≤ {reorder_pt} (lead time {lead_time} days)\n"
                f"  {item2}: starts with {i2_init} units, sells {i2_demand}/day, "
                f"reorders {reorder_qty} units when stock ≤ {reorder_pt} (lead time {lead_time} days)\n"
                f"What is the total number of units (both items combined) at the end of day {query_day}?"
            )
            distractors = [str(correct - 5), str(correct + 8), str(correct + 15)]
            distractors = [d for d in distractors if d != answer and int(d) >= 0]
            metadata = {"params": params, "correct_answer": correct, "mode": "two_items"}

        elif difficulty == 4:
            initial = rng.randint(60, 150)
            demand_w1 = rng.randint(5, 12)
            demand_w2 = rng.randint(10, 20)
            query_day = rng.randint(8, 18)
            params = {
                "mode": "seasonal",
                "initial": initial,
                "demand_week1": demand_w1,
                "demand_week2": demand_w2,
                "query_day": query_day,
            }
            correct = self._simulate(params)
            answer = str(correct)
            question = (
                f"A store tracks {product} inventory with seasonal demand.\n"
                f"Starting inventory: {initial} units.\n"
                f"Week 1 (days 1-7): sells {demand_w1} units per day.\n"
                f"Week 2+ (day 8 onward): sells {demand_w2} units per day.\n"
                f"Stock never goes below 0. How many units remain at the end of day {query_day}?"
            )
            distractors = [str(correct - 3), str(correct + 5), str(correct + 12)]
            distractors = [d for d in distractors if d != answer and int(d) >= 0]
            metadata = {"params": params, "correct_answer": correct, "mode": "seasonal"}

        else:  # difficulty 5
            wh_init = rng.randint(100, 200)
            s1_init = rng.randint(30, 70)
            s2_init = rng.randint(30, 70)
            wh_demand = rng.randint(5, 12)
            s1_demand = rng.randint(4, 9)
            s2_demand = rng.randint(4, 9)
            transfer_day = rng.randint(3, 7)
            transfer_qty = rng.randint(20, 50)
            query_day = rng.randint(8, 14)
            params = {
                "mode": "multi_echelon",
                "warehouse_initial": wh_init,
                "store1_initial": s1_init,
                "store2_initial": s2_init,
                "warehouse_demand": wh_demand,
                "store1_demand": s1_demand,
                "store2_demand": s2_demand,
                "transfer_day": transfer_day,
                "transfer_qty": transfer_qty,
                "query_day": query_day,
            }
            correct = self._simulate(params)
            answer = str(correct)
            question = (
                f"A supply chain has one central warehouse and two stores, all stocking {product}.\n"
                f"  Warehouse: starts with {wh_init} units, sells {wh_demand}/day to direct customers.\n"
                f"  Store 1: starts with {s1_init} units, sells {s1_demand}/day.\n"
                f"  Store 2: starts with {s2_init} units, sells {s2_demand}/day.\n"
                f"On day {transfer_day}, the warehouse transfers {transfer_qty} units equally to the two stores "
                f"(rounded down for Store 1, remainder to Store 2).\n"
                f"Stock never goes below 0. "
                f"What is the total system stock (warehouse + both stores) at the end of day {query_day}?"
            )
            distractors = [str(correct - 5), str(correct + 8), str(correct + 15)]
            distractors = [d for d in distractors if d != answer and int(d) >= 0]
            metadata = {"params": params, "correct_answer": correct, "mode": "multi_echelon"}

        return TaskEntry(
            question=question,
            answer=answer,
            distractors=distractors,
            difficulty=difficulty,
            domain=self.domain,
            metadata=metadata,
            task_id="",
        )

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        try:
            proposed = int(answer.strip().replace(",", ""))
            return 1.0 if proposed == entry.metadata["correct_answer"] else 0.0
        except (ValueError, KeyError):
            return 0.0
