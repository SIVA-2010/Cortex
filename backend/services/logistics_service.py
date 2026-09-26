from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class LogisticsService:
    REQUIRED_FILES = ("orders.csv", "products.csv", "carriers.json")
    ORDER_COLUMNS = {
        "order_id",
        "customer_id",
        "sku",
        "qty",
        "order_date",
        "promised_date",
        "ship_to_region",
        "status",
    }
    PRODUCT_COLUMNS = {
        "sku",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "unit_price",
        "weight_kg",
    }

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "sample_data" / "logistics"

    @classmethod
    def readiness(cls, data_dir: str | Path | None = None) -> dict[str, Any]:
        directory = Path(data_dir or cls.default_data_dir()).resolve()
        present = [name for name in cls.REQUIRED_FILES if (directory / name).is_file()]
        missing = [name for name in cls.REQUIRED_FILES if name not in present]
        return {
            "ready": not missing,
            "data_dir": str(directory),
            "required_files": list(cls.REQUIRED_FILES),
            "present_files": present,
            "missing_files": missing,
        }

    @classmethod
    def validate_data_package(cls, data_dir: str | Path | None = None) -> Path:
        readiness = cls.readiness(data_dir)
        if not readiness["ready"]:
            missing = ", ".join(readiness["missing_files"])
            raise ValueError(
                "Logistics Data Required — add orders.csv, products.csv and carriers.json "
                f"to {readiness['data_dir']}. Missing: {missing}."
            )
        directory = Path(readiness["data_dir"])
        orders = pd.read_csv(directory / "orders.csv")
        products = pd.read_csv(directory / "products.csv")
        with (directory / "carriers.json").open("r", encoding="utf-8") as handle:
            carriers = json.load(handle)

        missing_orders = sorted(cls.ORDER_COLUMNS - set(orders.columns))
        missing_products = sorted(cls.PRODUCT_COLUMNS - set(products.columns))
        if missing_orders:
            raise ValueError(
                "Dataset Mismatch — orders.csv is missing required columns: " + ", ".join(missing_orders)
            )
        if missing_products:
            raise ValueError(
                "Dataset Mismatch — products.csv is missing required columns: " + ", ".join(missing_products)
            )
        if not isinstance(carriers, list) or not carriers:
            raise ValueError("Dataset Mismatch — carriers.json must contain a non-empty list of carriers.")
        required_carrier = {
            "carrier_id",
            "carrier_name",
            "service_level",
            "base_cost",
            "cost_per_kg",
            "eta_days",
            "regions_covered",
            "reliability",
        }
        for index, carrier in enumerate(carriers, start=1):
            if not isinstance(carrier, dict):
                raise ValueError(f"Dataset Mismatch — carrier record {index} is not an object.")
            missing = sorted(required_carrier - set(carrier))
            if missing:
                raise ValueError(
                    f"Dataset Mismatch — carrier record {index} is missing: {', '.join(missing)}"
                )

            if not str(carrier.get("carrier_id", "")).strip() or not str(carrier.get("carrier_name", "")).strip():
                raise ValueError(
                    f"Dataset Mismatch — carrier record {index} must contain a carrier_id and carrier_name."
                )
            try:
                base_cost = float(carrier["base_cost"])
                cost_per_kg = float(carrier["cost_per_kg"])
                eta_days = int(carrier["eta_days"])
                reliability = float(carrier["reliability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dataset Mismatch — carrier record {index} contains invalid cost, ETA or reliability values."
                ) from exc
            if base_cost < 0 or cost_per_kg < 0 or eta_days <= 0 or not 0.0 <= reliability <= 1.0:
                raise ValueError(
                    f"Dataset Mismatch — carrier record {index} has out-of-range cost, ETA or reliability values."
                )
            if not str(carrier.get("regions_covered", "")).strip():
                raise ValueError(
                    f"Dataset Mismatch — carrier record {index} has no covered regions."
                )

        if orders.empty:
            raise ValueError("Dataset Mismatch — orders.csv contains no order rows.")
        if products.empty:
            raise ValueError("Dataset Mismatch — products.csv contains no product rows.")

        order_ids = orders["order_id"].fillna("").astype(str).str.strip()
        skus = orders["sku"].fillna("").astype(str).str.strip()
        regions = orders["ship_to_region"].fillna("").astype(str).str.strip()
        quantities = pd.to_numeric(orders["qty"], errors="coerce")
        order_dates = pd.to_datetime(orders["order_date"], errors="coerce")
        promised_dates = pd.to_datetime(orders["promised_date"], errors="coerce")

        usable_orders = (
            order_ids.ne("")
            & skus.ne("")
            & regions.ne("")
            & quantities.notna()
            & quantities.gt(0)
            & order_dates.notna()
            & promised_dates.notna()
        )
        usable_ratio = float(usable_orders.mean()) if len(orders) else 0.0
        if usable_ratio < 0.70:
            raise ValueError(
                "Dataset Mismatch — orders.csv does not contain enough usable logistics rows. "
                "At least 70% of rows must have order ID, SKU, positive quantity, valid dates and region."
            )

        duplicate_ratio = float(order_ids[usable_orders].duplicated(keep=False).mean()) if usable_orders.any() else 0.0
        if duplicate_ratio > 0.25:
            raise ValueError(
                "Dataset Mismatch — too many usable order rows reuse the same order_id."
            )

        pending_mask = orders["status"].fillna("").astype(str).str.strip().str.lower().eq("pending") & usable_orders
        if not pending_mask.any():
            raise ValueError(
                "Dataset Not Actionable — orders.csv contains no usable pending orders for the current logistics objective."
            )

        product_skus = products["sku"].fillna("").astype(str).str.strip()
        product_weights = pd.to_numeric(products["weight_kg"], errors="coerce")
        product_prices = pd.to_numeric(products["unit_price"], errors="coerce")
        usable_products = (
            product_skus.ne("")
            & products["product_name"].fillna("").astype(str).str.strip().ne("")
            & product_weights.notna()
            & product_weights.gt(0)
            & product_prices.notna()
            & product_prices.ge(0)
        )
        if float(usable_products.mean()) < 0.70:
            raise ValueError(
                "Dataset Mismatch — products.csv does not contain enough usable product/SKU, price and weight data."
            )

        valid_product_skus = set(product_skus[usable_products])
        pending_skus = skus[pending_mask]
        matched_product_ratio = float(pending_skus.isin(valid_product_skus).mean()) if len(pending_skus) else 0.0
        if matched_product_ratio < 0.50:
            raise ValueError(
                "Dataset Mismatch — most pending order SKUs do not exist in products.csv. "
                "The orders and products files do not appear to belong to the same logistics dataset."
            )

        covered_regions: set[str] = set()
        for carrier in carriers:
            covered_regions.update(
                value.strip().lower()
                for value in str(carrier["regions_covered"]).split(",")
                if value.strip()
            )
        pending_regions = regions[pending_mask].str.lower()
        covered_ratio = float(pending_regions.isin(covered_regions).mean()) if len(pending_regions) else 0.0
        if covered_ratio == 0.0:
            raise ValueError(
                "Dataset Mismatch — no carrier in carriers.json covers any pending-order region. "
                "The carrier file does not appear compatible with orders.csv."
            )

        return directory

    @staticmethod
    def detect_optimization_mode(objective: str) -> str:
        text = str(objective or "").lower()
        if any(term in text for term in ("fastest", "quickest", "speed", "urgent")):
            return "fastest"
        if any(term in text for term in ("cheapest", "lowest cost", "low cost", "minimise cost", "minimize cost")):
            return "cheapest"
        return "balanced"

    @staticmethod
    def _region_is_covered(carrier_regions: Any, order_region: Any) -> bool:
        if pd.isna(order_region):
            return False
        regions = {
            item.strip().lower()
            for item in str(carrier_regions).split(",")
            if item.strip()
        }
        return str(order_region).strip().lower() in regions

    @staticmethod
    def _shipping_cost(
        base_cost: float,
        cost_per_kg: float,
        weight_kg: float,
        qty: int,
    ) -> tuple[float, float]:
        total_weight = float(weight_kg) * int(qty)
        return round(float(base_cost) + float(cost_per_kg) * total_weight, 2), round(total_weight, 2)

    @staticmethod
    def _delay_risk(order_date: Any, promised_date: Any, eta_days: int) -> dict[str, str]:
        order_dt = pd.to_datetime(order_date, errors="coerce")
        promised_dt = pd.to_datetime(promised_date, errors="coerce")
        if pd.isna(order_dt) or pd.isna(promised_dt):
            return {"delivery_risk": "DATE_CHECK_FAILED", "expected_delivery_date": "UNKNOWN"}
        expected = order_dt + pd.Timedelta(days=int(eta_days))
        return {
            "delivery_risk": "DELAY_RISK" if expected > promised_dt else "ON_TIME",
            "expected_delivery_date": expected.date().isoformat(),
        }

    @classmethod
    def _carrier_rates(
        cls,
        carriers: list[dict[str, Any]],
        *,
        region: str,
        weight_kg: float,
        qty: int,
    ) -> list[dict[str, Any]]:
        rates: list[dict[str, Any]] = []
        for carrier in carriers:
            if not cls._region_is_covered(carrier["regions_covered"], region):
                continue
            cost, total_weight = cls._shipping_cost(
                float(carrier["base_cost"]),
                float(carrier["cost_per_kg"]),
                float(weight_kg),
                int(qty),
            )
            rates.append(
                {
                    "carrier_id": str(carrier["carrier_id"]),
                    "carrier_name": str(carrier["carrier_name"]),
                    "service_level": str(carrier["service_level"]),
                    "cost": cost,
                    "eta_days": int(carrier["eta_days"]),
                    "reliability": float(carrier["reliability"]),
                    "total_weight_kg": total_weight,
                    "regions_covered": str(carrier["regions_covered"]),
                }
            )
        return rates

    @staticmethod
    def _choose_best_carrier(
        rates: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any] | None:
        if not rates:
            return None
        if mode == "cheapest":
            return sorted(
                rates,
                key=lambda rate: (rate["cost"], rate["eta_days"], -rate["reliability"]),
            )[0]
        if mode == "fastest":
            return sorted(
                rates,
                key=lambda rate: (rate["eta_days"], rate["cost"], -rate["reliability"]),
            )[0]

        costs = [float(rate["cost"]) for rate in rates]
        etas = [float(rate["eta_days"]) for rate in rates]
        min_cost, max_cost = min(costs), max(costs)
        min_eta, max_eta = min(etas), max(etas)

        def normalize(value: float, minimum: float, maximum: float) -> float:
            return 0.0 if maximum == minimum else (value - minimum) / (maximum - minimum)

        scored: list[dict[str, Any]] = []
        for rate in rates:
            cost_score = normalize(float(rate["cost"]), min_cost, max_cost)
            eta_score = normalize(float(rate["eta_days"]), min_eta, max_eta)
            reliability_score = 1.0 - float(rate["reliability"])
            score = 0.40 * cost_score + 0.40 * eta_score + 0.20 * reliability_score
            item = dict(rate)
            item["optimization_score"] = round(score, 4)
            scored.append(item)
        return sorted(scored, key=lambda item: item["optimization_score"])[0]

    @classmethod
    def build_shipping_plan(
        cls,
        *,
        data_dir: str | Path | None = None,
        objective: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        directory = cls.validate_data_package(data_dir)
        orders = pd.read_csv(directory / "orders.csv")
        products = pd.read_csv(directory / "products.csv")
        with (directory / "carriers.json").open("r", encoding="utf-8") as handle:
            carriers: list[dict[str, Any]] = json.load(handle)

        mode = cls.detect_optimization_mode(objective)
        pending = orders[
            orders["status"].fillna("").astype(str).str.lower().eq("pending")
        ].copy()
        pending["_promised_sort"] = pd.to_datetime(pending["promised_date"], errors="coerce")
        pending = pending.sort_values("_promised_sort", na_position="last").drop(columns=["_promised_sort"])
        pending = pending.head(max(int(limit), 1))

        product_by_sku = {
            str(row["sku"]): row
            for _, row in products.iterrows()
        }
        plan: list[dict[str, Any]] = []
        exceptions: list[dict[str, str]] = []
        evidence_catalog: list[dict[str, Any]] = []

        for _, order in pending.iterrows():
            order_id = str(order["order_id"])
            sku = str(order["sku"])
            product = product_by_sku.get(sku)
            if product is None:
                exceptions.append({"order_id": order_id, "issue": f"Product not found for SKU {sku}."})
                continue
            try:
                qty = int(order["qty"])
                weight_kg = float(product["weight_kg"])
            except Exception:
                exceptions.append({"order_id": order_id, "issue": "Invalid quantity or product weight."})
                continue
            if qty <= 0 or weight_kg <= 0:
                exceptions.append({"order_id": order_id, "issue": "Quantity and product weight must be positive."})
                continue

            rates = cls._carrier_rates(
                carriers,
                region=str(order["ship_to_region"]),
                weight_kg=weight_kg,
                qty=qty,
            )
            chosen = cls._choose_best_carrier(rates, mode)
            if chosen is None:
                exceptions.append(
                    {
                        "order_id": order_id,
                        "issue": f"No carrier covers region {order['ship_to_region']}.",
                    }
                )
                continue

            risk = cls._delay_risk(order["order_date"], order["promised_date"], int(chosen["eta_days"]))
            approval_reasons: list[str] = []
            if risk["delivery_risk"] == "DELAY_RISK":
                approval_reasons.append("Carrier ETA may exceed the promised delivery date.")
            if float(chosen["total_weight_kg"]) > 100.0:
                approval_reasons.append("Shipment weight exceeds the 100 kg demo threshold.")
            if float(chosen["cost"]) > 5000.0:
                approval_reasons.append("Shipping cost exceeds the demo approval threshold.")

            plan_item = {
                "order_id": order_id,
                "customer_id": str(order["customer_id"]),
                "sku": sku,
                "product_name": str(product["product_name"]),
                "category": str(product["category"]),
                "qty": qty,
                "ship_to_region": str(order["ship_to_region"]),
                "order_date": str(order["order_date"]),
                "promised_date": str(order["promised_date"]),
                "expected_delivery_date": risk["expected_delivery_date"],
                "total_weight_kg": float(chosen["total_weight_kg"]),
                "chosen_carrier": str(chosen["carrier_name"]),
                "carrier_id": str(chosen["carrier_id"]),
                "service_level": str(chosen["service_level"]),
                "shipping_cost": float(chosen["cost"]),
                "eta_days": int(chosen["eta_days"]),
                "reliability": float(chosen["reliability"]),
                "tracking_id": (
                    "PENDING_MANAGER_APPROVAL"
                    if approval_reasons
                    else f"SIM-{order_id}-{chosen['carrier_id']}"
                ),
                "shipment_status": (
                    "PENDING_MANAGER_APPROVAL"
                    if approval_reasons
                    else "SIMULATED_CREATED"
                ),
                "delivery_risk": risk["delivery_risk"],
                "approval_required": bool(approval_reasons),
                "approval_reasons": approval_reasons,
                "optimization_mode": mode,
            }
            plan.append(plan_item)

            evidence_catalog.extend(
                [
                    {
                        "evidence_id": f"LOG-ORDER-{order_id}",
                        "evidence_type": "logistics_order",
                        "strength": "direct",
                        "order_id": order_id,
                        "summary": (
                            f"Order {order_id} for {sku}, quantity {qty}, region {order['ship_to_region']}, "
                            f"promised {order['promised_date']}."
                        ),
                    },
                    {
                        "evidence_id": f"LOG-PRODUCT-{sku}",
                        "evidence_type": "product_record",
                        "strength": "direct",
                        "sku": sku,
                        "summary": f"{product['product_name']} weighs {float(product['weight_kg']):.2f} kg per unit.",
                    },
                    {
                        "evidence_id": f"LOG-CARRIER-{chosen['carrier_id']}",
                        "evidence_type": "carrier_record",
                        "strength": "direct",
                        "carrier_id": str(chosen["carrier_id"]),
                        "summary": (
                            f"{chosen['carrier_name']} provides {chosen['service_level']} service with "
                            f"ETA {chosen['eta_days']} day(s), reliability {chosen['reliability']:.2f}, "
                            f"calculated cost {chosen['cost']:.2f}."
                        ),
                    },
                ]
            )

        deduped_evidence = {
            str(item["evidence_id"]): item for item in evidence_catalog
        }
        total_cost = round(sum(float(item["shipping_cost"]) for item in plan), 2)
        average_eta = round(
            sum(int(item["eta_days"]) for item in plan) / len(plan), 2
        ) if plan else 0.0
        return {
            "synthetic_data": True,
            "simulation_only": True,
            "data_dir": str(directory),
            "optimization_mode": mode,
            "pending_orders_available": int(len(orders[orders["status"].fillna("").astype(str).str.lower().eq("pending")])),
            "orders_planned": len(plan),
            "total_shipping_cost": total_cost,
            "average_eta_days": average_eta,
            "approval_required_count": sum(bool(item["approval_required"]) for item in plan),
            "shipping_plan": plan,
            "exceptions": exceptions,
            "evidence_catalog": list(deduped_evidence.values()),
            "optimization_weights": (
                {"cost": 0.40, "eta": 0.40, "reliability": 0.20}
                if mode == "balanced"
                else None
            ),
            "notice": "SIMULATED EXECUTION — no external carrier booking is performed.",
        }

    @staticmethod
    def build_assurance_payload(
        shipping_plan: dict[str, Any],
        *,
        source_agent_code: str,
    ) -> dict[str, Any]:
        plan = shipping_plan.get("shipping_plan", [])
        findings: list[dict[str, Any]] = []
        for index, item in enumerate(plan, start=1):
            order_id = str(item["order_id"])
            sku = str(item["sku"])
            carrier_id = str(item["carrier_id"])
            finding_id = f"LG-{index:03d}"
            delay_risk = str(item.get("delivery_risk", "UNKNOWN"))
            approval_required = bool(item.get("approval_required", False))
            findings.append(
                {
                    "finding_id": finding_id,
                    "source_agent_code": source_agent_code,
                    "title": f"Fulfilment decision for {order_id}",
                    "category": "Logistics Fulfilment",
                    "status": "confirmed",
                    "root_cause": (
                        f"Order {order_id} is assigned to {item['chosen_carrier']} "
                        f"({item['service_level']}) with a calculated shipping cost of "
                        f"{float(item['shipping_cost']):.2f}, ETA {int(item['eta_days'])} day(s), "
                        f"and delivery status {delay_risk}."
                    ),
                    "reasoning": (
                        f"Carrier selection used the {item['optimization_mode']} policy and only carriers "
                        "covering the destination region were considered."
                    ),
                    "supporting_evidence_ids": [
                        f"LOG-ORDER-{order_id}",
                        f"LOG-PRODUCT-{sku}",
                        f"LOG-CARRIER-{carrier_id}",
                    ],
                    "complaint_evidence_ids": [],
                    "confidence": min(0.98, 0.78 + float(item.get("reliability", 0.0)) * 0.20),
                    "business_impact": (
                        f"Planned simulated fulfilment cost {float(item['shipping_cost']):.2f}; "
                        f"expected delivery {item['expected_delivery_date']}."
                    ),
                    "recommended_action": (
                        f"Approve simulated fulfilment for {order_id} with {item['chosen_carrier']}."
                        if approval_required
                        else f"Use {item['chosen_carrier']} for the simulated fulfilment of {order_id}."
                    ),
                    "risk_level": "high" if approval_required else "low",
                    "human_approval_required": approval_required,
                    "approval_level": "reviewer" if approval_required else "none",
                    "evidence_gaps": [],
                    "simulation_only": True,
                }
            )

        cross_cutting_risks = [
            "Carrier cost, ETA and reliability values are synthetic demonstration inputs.",
            "Tracking and fulfilment actions are simulated; no external carrier system is called.",
        ]
        if shipping_plan.get("exceptions"):
            cross_cutting_risks.append(
                f"{len(shipping_plan['exceptions'])} order(s) could not be planned and require review."
            )

        return {
            "analysis": {
                "findings": findings,
                "cross_cutting_risks": cross_cutting_risks,
                "summary": {
                    "orders_planned": shipping_plan.get("orders_planned", 0),
                    "total_shipping_cost": shipping_plan.get("total_shipping_cost", 0.0),
                    "average_eta_days": shipping_plan.get("average_eta_days", 0.0),
                    "optimization_mode": shipping_plan.get("optimization_mode", "balanced"),
                },
            },
            "operational_evidence_summary": {
                "evidence_catalog": list(shipping_plan.get("evidence_catalog", [])),
                "source": "synthetic_logistics_package",
                "simulation_only": True,
            },
            "shipping_plan": shipping_plan,
            "synthetic_data": True,
            "simulation_only": True,
        }
