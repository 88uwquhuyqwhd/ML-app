from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / 'demo_datasets'


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def clip_probability(value):
    return np.clip(value, 0.01, 0.99)


def choice(rng, values, size, p=None):
    return rng.choice(values, size=size, p=p)


def sample_builder(builder: Callable[[np.random.Generator], pd.DataFrame], rows: int):
    def wrapped(rng: np.random.Generator):
        df = builder(rng)
        sample_rows = min(rows, len(df))
        sample_seed = int(rng.integers(0, 1_000_000))
        return df.sample(n=sample_rows, random_state=sample_seed).reset_index(drop=True)

    return wrapped


def add_missing(rng, df: pd.DataFrame, columns: list[str], rate: float):
    next_df = df.copy()
    for column in columns:
        mask = rng.random(len(next_df)) < rate
        next_df.loc[mask, column] = np.nan
    return next_df


def add_outliers(rng, df: pd.DataFrame, column: str, multiplier: float = 4.0, rate: float = 0.015):
    next_df = df.copy()
    mask = rng.random(len(next_df)) < rate
    if mask.any():
        next_df.loc[mask, column] = next_df.loc[mask, column] * multiplier
    return next_df


@dataclass
class DatasetSpec:
    folder: str
    filename: str
    module: str
    scenario: str
    recommended_usage: str
    builder: Callable[[np.random.Generator], pd.DataFrame]


def build_customer_churn(rng):
    n = 1800
    tenure = rng.integers(1, 72, n)
    monthly_fee = rng.normal(88, 22, n).clip(25, 220)
    support_calls = rng.poisson(1.8, n)
    data_usage_gb = rng.normal(130, 45, n).clip(8, 320)
    payment_delay_days = rng.poisson(2.4, n)
    satisfaction = rng.normal(6.3, 1.6, n).clip(1, 10)
    contract_type = choice(rng, ['month_to_month', 'one_year', 'two_year'], n, p=[0.52, 0.28, 0.20])
    internet_type = choice(rng, ['fiber', 'broadband', 'mobile'], n, p=[0.46, 0.36, 0.18])
    city_tier = choice(rng, ['tier_1', 'tier_2', 'tier_3'], n, p=[0.24, 0.43, 0.33])
    auto_pay = choice(rng, ['yes', 'no'], n, p=[0.57, 0.43])
    senior = choice(rng, ['yes', 'no'], n, p=[0.19, 0.81])
    score = (-1.3 - 0.025 * tenure + 0.020 * monthly_fee + 0.32 * support_calls + 0.07 * payment_delay_days - 0.24 * satisfaction + 0.55 * (contract_type == 'month_to_month') + 0.28 * (internet_type == 'fiber') + 0.35 * (auto_pay == 'no') + 0.16 * (senior == 'yes') + rng.normal(0, 0.45, n))
    churn = np.where(rng.random(n) < sigmoid(score), 'yes', 'no')
    return pd.DataFrame({'tenure_months': tenure, 'monthly_fee': monthly_fee.round(2), 'support_calls_90d': support_calls, 'data_usage_gb': data_usage_gb.round(2), 'payment_delay_days': payment_delay_days, 'satisfaction_score': satisfaction.round(2), 'contract_type': contract_type, 'internet_type': internet_type, 'city_tier': city_tier, 'auto_pay': auto_pay, 'senior_customer': senior, 'churn': churn})


def build_credit_risk(rng):
    n = 2200
    income = rng.normal(18, 7, n).clip(3.5, 55)
    debt_ratio = rng.normal(0.38, 0.18, n).clip(0.02, 1.15)
    loan_amount = rng.normal(22, 10, n).clip(2, 70)
    credit_history = rng.integers(3, 240, n)
    overdue_count = rng.poisson(0.9, n)
    employment_years = rng.normal(6.4, 4.5, n).clip(0, 35)
    home_ownership = choice(rng, ['rent', 'mortgage', 'own'], n, p=[0.44, 0.41, 0.15])
    loan_purpose = choice(rng, ['working_capital', 'car', 'education', 'medical', 'consumer'], n)
    region = choice(rng, ['east', 'north', 'south', 'west'], n)
    score = (-1.7 + 2.2 * debt_ratio + 0.035 * loan_amount + 0.34 * overdue_count - 0.055 * income - 0.010 * credit_history - 0.045 * employment_years + 0.36 * (home_ownership == 'rent') + 0.18 * (loan_purpose == 'consumer') + rng.normal(0, 0.42, n))
    default_risk = np.where(rng.random(n) < sigmoid(score), 'high', 'low')
    return pd.DataFrame({'monthly_income_k': income.round(2), 'debt_ratio': debt_ratio.round(3), 'loan_amount_k': loan_amount.round(2), 'credit_history_months': credit_history, 'overdue_count': overdue_count, 'employment_years': employment_years.round(1), 'home_ownership': home_ownership, 'loan_purpose': loan_purpose, 'region': region, 'default_risk': default_risk})


def build_hr_attrition(rng):
    n = 1500
    years_at_company = rng.integers(0, 25, n)
    salary_k = rng.normal(17, 5.5, n).clip(5, 42)
    overtime_hours = rng.normal(8, 6, n).clip(0, 36)
    manager_rating = rng.normal(3.5, 0.7, n).clip(1, 5)
    commute_km = rng.normal(11, 6, n).clip(0.5, 45)
    work_life_balance = rng.normal(3.2, 0.8, n).clip(1, 5)
    department = choice(rng, ['sales', 'it', 'hr', 'finance', 'operations'], n, p=[0.24, 0.27, 0.11, 0.14, 0.24])
    travel_freq = choice(rng, ['rare', 'often', 'frequent'], n, p=[0.50, 0.33, 0.17])
    promotion_gap = rng.integers(0, 8, n)
    score = (-1.4 - 0.08 * years_at_company - 0.07 * salary_k + 0.09 * overtime_hours - 0.40 * manager_rating + 0.04 * commute_km - 0.32 * work_life_balance + 0.24 * promotion_gap + 0.28 * (travel_freq == 'frequent') + 0.19 * (department == 'sales') + rng.normal(0, 0.45, n))
    attrition = np.where(rng.random(n) < sigmoid(score), 'leave', 'stay')
    return pd.DataFrame({'years_at_company': years_at_company, 'salary_k': salary_k.round(2), 'overtime_hours_week': overtime_hours.round(1), 'manager_rating': manager_rating.round(2), 'commute_km': commute_km.round(1), 'work_life_balance': work_life_balance.round(2), 'department': department, 'travel_frequency': travel_freq, 'promotion_gap_years': promotion_gap, 'attrition': attrition})


def build_medical_screening(rng):
    n = 1600
    age = rng.integers(18, 83, n)
    bmi = rng.normal(25.8, 4.8, n).clip(16, 42)
    systolic_bp = rng.normal(124, 17, n).clip(88, 198)
    glucose = rng.normal(101, 24, n).clip(62, 230)
    exercise = rng.normal(3.2, 1.8, n).clip(0, 10)
    symptom_score = rng.normal(4.8, 2.6, n).clip(0, 10)
    smoking_status = choice(rng, ['never', 'former', 'current'], n, p=[0.51, 0.28, 0.21])
    family_history = choice(rng, ['yes', 'no'], n, p=[0.39, 0.61])
    sleep_hours = rng.normal(7.0, 1.1, n).clip(4, 10)
    score = (-4.2 + 0.028 * age + 0.082 * bmi + 0.018 * systolic_bp + 0.022 * glucose - 0.18 * exercise + 0.35 * symptom_score + 0.62 * (family_history == 'yes') + 0.44 * (smoking_status == 'current') - 0.10 * sleep_hours + rng.normal(0, 0.48, n))
    diagnosis = np.where(rng.random(n) < sigmoid(score), 'positive', 'negative')
    return pd.DataFrame({'age': age, 'bmi': bmi.round(2), 'systolic_bp': systolic_bp.round(1), 'glucose': glucose.round(1), 'exercise_hours_week': exercise.round(1), 'symptom_score': symptom_score.round(1), 'smoking_status': smoking_status, 'family_history': family_history, 'sleep_hours': sleep_hours.round(1), 'screening_result': diagnosis})


def build_marketing_response(rng):
    n = 2100
    previous_clicks = rng.poisson(4.6, n)
    exposures = rng.poisson(17, n) + 1
    discount_rate = rng.normal(0.14, 0.06, n).clip(0, 0.35)
    last_purchase_days = rng.integers(1, 365, n)
    tenure_months = rng.integers(1, 84, n)
    spend_score = rng.normal(58, 18, n).clip(5, 100)
    channel = choice(rng, ['sms', 'email', 'app_push', 'ads'], n, p=[0.19, 0.34, 0.28, 0.19])
    segment = choice(rng, ['new', 'growth', 'vip', 'sleeping'], n, p=[0.21, 0.33, 0.20, 0.26])
    region = choice(rng, ['east', 'north', 'south', 'west'], n)
    score = (-2.7 + 0.08 * previous_clicks + 0.05 * exposures + 2.8 * discount_rate - 0.006 * last_purchase_days + 0.012 * tenure_months + 0.018 * spend_score + 0.33 * (channel == 'app_push') + 0.42 * (segment == 'vip') - 0.28 * (segment == 'sleeping') + rng.normal(0, 0.52, n))
    responded = np.where(rng.random(n) < sigmoid(score), 'yes', 'no')
    return pd.DataFrame({'previous_clicks': previous_clicks, 'campaign_exposures': exposures, 'discount_rate': discount_rate.round(3), 'last_purchase_days': last_purchase_days, 'tenure_months': tenure_months, 'spend_score': spend_score.round(1), 'channel': channel, 'segment': segment, 'region': region, 'responded': responded})


def build_student_dropout(rng):
    n = 1500
    attendance = rng.normal(82, 11, n).clip(30, 100)
    gpa = rng.normal(2.95, 0.58, n).clip(0.8, 4.0)
    study_hours = rng.normal(12, 6, n).clip(0, 36)
    failed_courses = rng.poisson(1.2, n)
    family_support = rng.normal(6.4, 1.9, n).clip(1, 10)
    part_time_job = choice(rng, ['yes', 'no'], n, p=[0.39, 0.61])
    scholarship = choice(rng, ['yes', 'no'], n, p=[0.27, 0.73])
    program = choice(rng, ['business', 'engineering', 'arts', 'science'], n)
    commute_minutes = rng.normal(38, 21, n).clip(5, 150)
    score = (-1.7 - 0.042 * attendance - 1.10 * gpa - 0.040 * study_hours + 0.38 * failed_courses - 0.17 * family_support + 0.28 * (part_time_job == 'yes') - 0.35 * (scholarship == 'yes') + 0.004 * commute_minutes + rng.normal(0, 0.46, n))
    dropout = np.where(rng.random(n) < sigmoid(score), 'yes', 'no')
    return pd.DataFrame({'attendance_rate': attendance.round(1), 'gpa': gpa.round(2), 'study_hours_week': study_hours.round(1), 'failed_courses': failed_courses, 'family_support_score': family_support.round(1), 'part_time_job': part_time_job, 'scholarship': scholarship, 'program': program, 'commute_minutes': commute_minutes.round(0), 'dropout': dropout})


def build_insurance_claim_fraud(rng):
    n = 2200
    claim_amount = rng.normal(18, 9, n).clip(0.8, 85)
    claim_delay_days = rng.poisson(5.5, n) + 1
    policy_tenure = rng.integers(1, 121, n)
    prior_claims = rng.poisson(0.8, n)
    vehicle_age = rng.integers(0, 18, n)
    premium_delay_days = rng.poisson(3.2, n)
    accident_severity = choice(rng, ['minor', 'moderate', 'major'], n, p=[0.47, 0.38, 0.15])
    repair_shop_type = choice(rng, ['network', 'outside'], n, p=[0.71, 0.29])
    police_report = choice(rng, ['yes', 'no'], n, p=[0.76, 0.24])
    night_accident = choice(rng, ['yes', 'no'], n, p=[0.27, 0.73])
    province = choice(rng, ['east', 'north', 'south', 'west'], n)
    score = (
        -4.2
        + 0.055 * claim_amount
        + 0.12 * claim_delay_days
        - 0.012 * policy_tenure
        + 0.24 * prior_claims
        + 0.05 * vehicle_age
        + 0.14 * premium_delay_days
        + 0.72 * (repair_shop_type == 'outside')
        + 0.62 * (police_report == 'no')
        + 0.42 * (night_accident == 'yes')
        + 0.55 * (accident_severity == 'minor')
        - 0.30 * (accident_severity == 'major')
        + rng.normal(0, 0.52, n)
    )
    fraud_label = np.where(rng.random(n) < sigmoid(score), 'fraud', 'legit')
    return pd.DataFrame({
        'claim_amount_k': claim_amount.round(2),
        'claim_delay_days': claim_delay_days,
        'policy_tenure_months': policy_tenure,
        'prior_claims': prior_claims,
        'vehicle_age_years': vehicle_age,
        'premium_delay_days': premium_delay_days,
        'accident_severity': accident_severity,
        'repair_shop_type': repair_shop_type,
        'police_report': police_report,
        'night_accident': night_accident,
        'province': province,
        'fraud_label': fraud_label,
    })


def build_equipment_fault_multiclass(rng):
    n = 1900
    temperature = rng.normal(76, 11, n).clip(42, 126)
    vibration = rng.normal(2.0, 0.8, n).clip(0.4, 5.6)
    pressure = rng.normal(32, 6.5, n).clip(15, 58)
    runtime_hours = rng.normal(14, 4.1, n).clip(2, 24)
    maintenance_gap = rng.normal(26, 12, n).clip(1, 90)
    defect_rate_prev = rng.normal(2.8, 1.5, n).clip(0.1, 12)
    operator_level = choice(rng, ['junior', 'mid', 'senior'], n, p=[0.28, 0.49, 0.23])
    line_id = choice(rng, ['L1', 'L2', 'L3', 'L4'], n)
    risk_score = (
        0.05 * (temperature - 70)
        + 0.90 * (vibration - 1.6)
        + 0.03 * (pressure - 30)
        + 0.06 * (runtime_hours - 12)
        + 0.03 * maintenance_gap
        + 0.25 * defect_rate_prev
        + 0.40 * (operator_level == 'junior')
        - 0.18 * (operator_level == 'senior')
        + rng.normal(0, 0.55, n)
    )
    equipment_status = np.select(
        [risk_score < 2.1, risk_score < 3.2],
        ['normal', 'warning'],
        default='failure',
    )
    return pd.DataFrame({
        'temperature_c': temperature.round(2),
        'vibration_level': vibration.round(3),
        'pressure_bar': pressure.round(2),
        'runtime_hours_day': runtime_hours.round(1),
        'maintenance_gap_days': maintenance_gap.round(0),
        'previous_defect_rate_pct': defect_rate_prev.round(2),
        'operator_level': operator_level,
        'line_id': line_id,
        'equipment_status': equipment_status,
    })


def build_app_subscription_upgrade_cn(rng):
    n = 1800
    login_days = rng.integers(1, 31, n)
    core_feature_usage = rng.poisson(22, n) + 1
    team_size = rng.lognormal(mean=2.2, sigma=0.8, size=n).clip(1, 180)
    current_plan = choice(rng, ['免费版', '基础版', '专业版'], n, p=[0.42, 0.38, 0.20])
    support_tickets = rng.poisson(1.7, n)
    active_hours = rng.normal(18, 9, n).clip(1, 85)
    automation_flow_count = rng.poisson(4.5, n)
    industry = choice(rng, ['电商', '教育', '制造', '咨询服务', '医疗'], n)
    data_volume = rng.normal(18, 11, n).clip(0.5, 90)
    score = (
        0.18 * login_days
        + 0.06 * core_feature_usage
        + 0.012 * team_size
        + 0.03 * active_hours
        + 0.09 * automation_flow_count
        + 0.04 * data_volume
        - 0.22 * support_tickets
        + 0.85 * (current_plan == '免费版')
        + 0.22 * (current_plan == '基础版')
        - 0.60 * (current_plan == '专业版')
        + rng.normal(0, 0.65, n)
    )
    upgrade_advice = np.select(
        [score < 7.0, score < 9.7],
        ['保留当前套餐', '升级专业版'],
        default='升级企业版',
    )
    return pd.DataFrame({
        '近30天登录天数': login_days,
        '核心功能使用次数': core_feature_usage,
        '团队成员数': np.round(team_size).astype(int),
        '当前套餐': current_plan,
        '客服咨询次数': support_tickets,
        '月活跃时长_小时': active_hours.round(1),
        '自动化流程数': automation_flow_count,
        '所属行业': industry,
        '数据量_GB': data_volume.round(2),
        '升级建议': upgrade_advice,
    })


def build_housing_price(rng):
    n = 1800
    area = rng.normal(106, 36, n).clip(35, 280)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    floor = rng.integers(1, 33, n)
    building_age = rng.integers(0, 35, n)
    distance_center = rng.normal(9, 5, n).clip(0.5, 28)
    school_score = rng.normal(7.2, 1.1, n).clip(4, 10)
    renovated = choice(rng, ['yes', 'no'], n, p=[0.31, 0.69])
    district = choice(rng, ['core', 'inner_ring', 'suburb'], n, p=[0.23, 0.42, 0.35])
    near_subway = choice(rng, ['yes', 'no'], n, p=[0.61, 0.39])
    price = (52 + 1.55 * area + 18 * bedrooms + 9 * bathrooms + 0.9 * floor - 1.1 * building_age - 4.8 * distance_center + 12 * school_score + 26 * (renovated == 'yes') + 78 * (district == 'core') + 32 * (district == 'inner_ring') + 25 * (near_subway == 'yes') + rng.normal(0, 18, n))
    return pd.DataFrame({'area_sqm': area.round(1), 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'floor': floor, 'building_age': building_age, 'distance_to_center_km': distance_center.round(2), 'school_score': school_score.round(1), 'renovated': renovated, 'district': district, 'near_subway': near_subway, 'price_k': price.round(2)})


def build_sales_forecast(rng):
    n = 1700
    ad_spend = rng.normal(35, 13, n).clip(3, 90)
    web_traffic = rng.normal(12000, 3800, n).clip(1200, 28000)
    discount_rate = rng.normal(0.11, 0.05, n).clip(0, 0.32)
    holiday_flag = choice(rng, [0, 1], n, p=[0.74, 0.26])
    competitor_price_index = rng.normal(1.0, 0.12, n).clip(0.72, 1.35)
    region = choice(rng, ['east', 'north', 'south', 'west'], n)
    product_tier = choice(rng, ['basic', 'standard', 'premium'], n, p=[0.28, 0.47, 0.25])
    month = rng.integers(1, 13, n)
    sales_units = (260 + 6.2 * ad_spend + 0.048 * web_traffic + 560 * discount_rate + 140 * holiday_flag - 170 * competitor_price_index + 85 * (product_tier == 'premium') + 35 * (region == 'east') + 16 * np.sin(month / 12 * 2 * np.pi) + rng.normal(0, 45, n))
    return pd.DataFrame({'month': month, 'ad_spend_k': ad_spend.round(2), 'web_traffic': web_traffic.round(0), 'discount_rate': discount_rate.round(3), 'holiday_flag': holiday_flag, 'competitor_price_index': competitor_price_index.round(3), 'region': region, 'product_tier': product_tier, 'sales_units': sales_units.round(1)})


def build_manufacturing_quality(rng):
    n = 1600
    line_speed = rng.normal(78, 10, n).clip(40, 115)
    temperature = rng.normal(186, 9, n).clip(150, 220)
    humidity = rng.normal(52, 11, n).clip(18, 85)
    vibration = rng.normal(1.8, 0.6, n).clip(0.5, 4.2)
    material_grade = choice(rng, ['A', 'B', 'C'], n, p=[0.48, 0.37, 0.15])
    shift = choice(rng, ['day', 'swing', 'night'], n, p=[0.41, 0.34, 0.25])
    machine_age = rng.integers(1, 18, n)
    operator_experience = rng.normal(5.8, 3.2, n).clip(0.5, 20)
    quality_score = (88 - 0.22 * line_speed - 0.18 * np.abs(temperature - 185) - 0.08 * humidity - 5.6 * vibration + 6.5 * (material_grade == 'A') + 2.8 * (material_grade == 'B') - 0.65 * machine_age + 0.7 * operator_experience - 2.2 * (shift == 'night') + rng.normal(0, 3.2, n))
    return pd.DataFrame({'line_speed': line_speed.round(2), 'temperature_c': temperature.round(2), 'humidity_pct': humidity.round(2), 'vibration_level': vibration.round(3), 'material_grade': material_grade, 'shift': shift, 'machine_age_years': machine_age, 'operator_experience_years': operator_experience.round(1), 'quality_score': quality_score.round(2)})


def build_energy_consumption(rng):
    n = 1900
    building_area = rng.normal(2400, 850, n).clip(400, 6200)
    occupancy = rng.integers(10, 320, n)
    outside_temp = rng.normal(18, 10, n).clip(-8, 38)
    equipment_hours = rng.normal(14, 4.5, n).clip(4, 24)
    insulation_level = choice(rng, ['low', 'medium', 'high'], n, p=[0.28, 0.46, 0.26])
    building_type = choice(rng, ['office', 'retail', 'school', 'hospital'], n)
    tariff_period = choice(rng, ['peak', 'flat', 'off_peak'], n, p=[0.31, 0.43, 0.26])
    humidity = rng.normal(53, 10, n).clip(18, 82)
    energy_kwh = (180 + 0.58 * building_area + 3.2 * occupancy + 12 * np.abs(outside_temp - 22) + 44 * equipment_hours + 95 * (building_type == 'hospital') + 48 * (building_type == 'retail') - 110 * (insulation_level == 'high') - 45 * (insulation_level == 'medium') + 62 * (tariff_period == 'peak') + 0.7 * humidity + rng.normal(0, 85, n))
    return pd.DataFrame({'building_area_sqm': building_area.round(1), 'occupancy': occupancy, 'outside_temp_c': outside_temp.round(1), 'equipment_hours_day': equipment_hours.round(1), 'humidity_pct': humidity.round(1), 'insulation_level': insulation_level, 'building_type': building_type, 'tariff_period': tariff_period, 'energy_kwh': energy_kwh.round(2)})


def build_ecommerce_order_value(rng):
    n = 2100
    session_duration = rng.normal(11.5, 5.4, n).clip(1, 40)
    pages_viewed = rng.poisson(9, n) + 1
    past_orders = rng.poisson(4.5, n)
    discount_rate = rng.normal(0.09, 0.05, n).clip(0, 0.28)
    loyalty_score = rng.normal(61, 19, n).clip(5, 100)
    source = choice(rng, ['organic', 'ads', 'social', 'email'], n, p=[0.32, 0.29, 0.19, 0.20])
    membership = choice(rng, ['none', 'silver', 'gold'], n, p=[0.46, 0.35, 0.19])
    device = choice(rng, ['mobile', 'desktop', 'tablet'], n, p=[0.58, 0.32, 0.10])
    cart_items = rng.poisson(3.4, n) + 1
    order_value = (18 + 1.8 * session_duration + 3.1 * pages_viewed + 2.6 * past_orders + 78 * discount_rate + 0.42 * loyalty_score + 4.5 * cart_items + 18 * (membership == 'gold') + 7 * (membership == 'silver') + 5 * (source == 'email') + rng.normal(0, 12, n))
    return pd.DataFrame({'session_duration_min': session_duration.round(2), 'pages_viewed': pages_viewed, 'past_orders': past_orders, 'discount_rate': discount_rate.round(3), 'loyalty_score': loyalty_score.round(1), 'traffic_source': source, 'membership_tier': membership, 'device_type': device, 'cart_items': cart_items, 'order_value': order_value.round(2)})


def build_logistics_delivery_time(rng):
    n = 1700
    distance = rng.normal(22, 12, n).clip(2, 85)
    weight = rng.normal(7.5, 4.8, n).clip(0.2, 30)
    weather_index = rng.normal(48, 19, n).clip(0, 100)
    route_complexity = rng.normal(5.5, 2.1, n).clip(1, 10)
    warehouse_load = rng.normal(63, 17, n).clip(15, 100)
    driver_experience = rng.normal(5.4, 3.2, n).clip(0.2, 20)
    vehicle_type = choice(rng, ['bike', 'van', 'truck'], n, p=[0.28, 0.53, 0.19])
    city_type = choice(rng, ['urban', 'suburban', 'county'], n, p=[0.52, 0.31, 0.17])
    delivery_hours = (0.8 + 0.11 * distance + 0.06 * weight + 0.012 * weather_index + 0.18 * route_complexity + 0.015 * warehouse_load - 0.055 * driver_experience + 0.30 * (vehicle_type == 'bike') + 0.22 * (city_type == 'urban') + rng.normal(0, 0.55, n))
    return pd.DataFrame({'distance_km': distance.round(2), 'package_weight_kg': weight.round(2), 'weather_index': weather_index.round(1), 'route_complexity': route_complexity.round(1), 'warehouse_load_pct': warehouse_load.round(1), 'driver_experience_years': driver_experience.round(1), 'vehicle_type': vehicle_type, 'city_type': city_type, 'delivery_hours': delivery_hours.round(2)})


def build_medical_cost_regression(rng):
    n = 2000
    age = rng.integers(18, 81, n)
    bmi = rng.normal(26.4, 4.9, n).clip(16, 43)
    chronic_conditions = rng.poisson(1.1, n).clip(0, 5)
    smoker = choice(rng, ['yes', 'no'], n, p=[0.24, 0.76])
    exercise_hours = rng.normal(3.4, 1.8, n).clip(0, 12)
    inpatient_days = rng.poisson(1.3, n)
    city_tier = choice(rng, ['tier_1', 'tier_2', 'tier_3'], n, p=[0.24, 0.41, 0.35])
    insurance_plan = choice(rng, ['basic', 'standard', 'premium'], n, p=[0.39, 0.42, 0.19])
    annual_checkups = rng.poisson(1.4, n)
    annual_cost = (
        2.8
        + 0.11 * age
        + 0.30 * bmi
        + 4.9 * chronic_conditions
        + 6.4 * (smoker == 'yes')
        + 1.55 * inpatient_days
        - 0.42 * exercise_hours
        + 2.1 * (city_tier == 'tier_1')
        + 3.0 * (insurance_plan == 'premium')
        + 1.2 * (insurance_plan == 'standard')
        + 0.45 * annual_checkups
        + rng.normal(0, 3.6, n)
    )
    return pd.DataFrame({
        'age': age,
        'bmi': bmi.round(2),
        'chronic_conditions': chronic_conditions,
        'smoker': smoker,
        'exercise_hours_week': exercise_hours.round(1),
        'inpatient_days_year': inpatient_days,
        'city_tier': city_tier,
        'insurance_plan': insurance_plan,
        'annual_checkups': annual_checkups,
        'annual_medical_cost_k': annual_cost.round(2),
    })


def build_store_revenue_regression_cn(rng):
    n = 1600
    store_area = rng.normal(165, 58, n).clip(35, 420)
    foot_traffic = rng.normal(520, 180, n).clip(60, 1500)
    staff_count = rng.integers(4, 42, n)
    promo_spend = rng.normal(5.5, 2.4, n).clip(0.2, 18)
    online_ratio = rng.normal(0.27, 0.14, n).clip(0.01, 0.85)
    business_district = choice(rng, ['核心商圈', '社区商圈', '校园商圈', '交通枢纽'], n, p=[0.25, 0.33, 0.18, 0.24])
    city_level = choice(rng, ['一线', '新一线', '二线', '三线'], n, p=[0.18, 0.27, 0.31, 0.24])
    years_open = rng.integers(1, 16, n)
    member_ratio = rng.normal(0.42, 0.17, n).clip(0.03, 0.95)
    monthly_revenue = (
        12
        + 0.085 * store_area
        + 0.034 * foot_traffic
        + 0.88 * staff_count
        + 2.15 * promo_spend
        + 20 * online_ratio
        + 11 * member_ratio
        + 7.8 * (business_district == '核心商圈')
        + 5.0 * (business_district == '交通枢纽')
        + 8.5 * (city_level == '一线')
        + 5.4 * (city_level == '新一线')
        - 0.45 * years_open
        + rng.normal(0, 6.8, n)
    )
    return pd.DataFrame({
        '门店面积_平米': store_area.round(1),
        '日均客流量': foot_traffic.round(0),
        '员工人数': staff_count,
        '促销投入_千元': promo_spend.round(2),
        '线上订单占比': online_ratio.round(3),
        '商圈类型': business_district,
        '城市级别': city_level,
        '开业年限': years_open,
        '会员销售占比': member_ratio.round(3),
        '月营业额_万元': monthly_revenue.round(2),
    })


def build_retail_segmentation(rng):
    n = 2400
    segment = choice(rng, ['premium', 'family', 'digital', 'price_sensitive'], n, p=[0.18, 0.31, 0.24, 0.27])
    base = pd.DataFrame({'hidden_segment': segment})
    base['age'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(41, 8, n), rng.normal(36, 7, n), rng.normal(29, 6, n)], default=rng.normal(34, 8, n)).clip(18, 68)
    base['income_k'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(32, 8, n), rng.normal(22, 6, n), rng.normal(18, 5, n)], default=rng.normal(14, 4, n)).clip(4, 60)
    base['monthly_visits'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(5.5, 1.4, n), rng.normal(7.2, 1.8, n), rng.normal(9.1, 2.0, n)], default=rng.normal(6.0, 1.7, n)).clip(1, 16)
    base['avg_spend'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(520, 110, n), rng.normal(310, 75, n), rng.normal(180, 55, n)], default=rng.normal(140, 40, n)).clip(25, 900)
    base['online_ratio'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(0.48, 0.14, n), rng.normal(0.34, 0.11, n), rng.normal(0.76, 0.10, n)], default=rng.normal(0.41, 0.12, n)).clip(0.02, 0.98)
    base['coupon_usage'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(0.12, 0.08, n), rng.normal(0.28, 0.11, n), rng.normal(0.18, 0.09, n)], default=rng.normal(0.45, 0.12, n)).clip(0, 0.9)
    base['return_rate'] = np.select([segment == 'premium', segment == 'family', segment == 'digital'], [rng.normal(0.08, 0.04, n), rng.normal(0.06, 0.03, n), rng.normal(0.12, 0.05, n)], default=rng.normal(0.09, 0.04, n)).clip(0, 0.45)
    return base.round(4)


def build_employee_factor(rng):
    n = 1000
    culture = rng.normal(0, 1, n)
    growth = rng.normal(0, 1, n)
    reward = rng.normal(0, 1, n)
    def likert(score):
        return np.round(np.clip(3 + score, 1, 5)).astype(int)
    return pd.DataFrame({
        'team_trust': likert(0.9 * culture + rng.normal(0, 0.5, n)),
        'manager_support': likert(0.8 * culture + rng.normal(0, 0.5, n)),
        'communication': likert(0.7 * culture + rng.normal(0, 0.6, n)),
        'process_clarity': likert(0.65 * culture + rng.normal(0, 0.6, n)),
        'promotion_fairness': likert(0.8 * growth + rng.normal(0, 0.5, n)),
        'learning_opportunity': likert(0.9 * growth + rng.normal(0, 0.5, n)),
        'career_path': likert(0.78 * growth + rng.normal(0, 0.6, n)),
        'skill_training': likert(0.72 * growth + rng.normal(0, 0.6, n)),
        'salary_competitiveness': likert(0.88 * reward + rng.normal(0, 0.5, n)),
        'bonus_satisfaction': likert(0.86 * reward + rng.normal(0, 0.5, n)),
        'benefit_satisfaction': likert(0.76 * reward + rng.normal(0, 0.6, n)),
        'recognition': likert(0.69 * reward + rng.normal(0, 0.6, n)),
        'department': choice(rng, ['sales', 'it', 'finance', 'operations', 'hr'], n),
    })


def build_ab_test(rng):
    n = 2400
    group = choice(rng, ['A', 'B'], n, p=[0.5, 0.5])
    region = choice(rng, ['east', 'north', 'south', 'west'], n)
    device = choice(rng, ['mobile', 'desktop'], n, p=[0.68, 0.32])
    session_seconds = rng.normal(210, 85, n).clip(30, 900)
    impressions = rng.poisson(14, n) + 1
    clicks = np.maximum(0, np.round(impressions * clip_probability(0.07 + 0.01 * (group == 'B') + 0.004 * (device == 'desktop') + rng.normal(0, 0.02, n)))).astype(int)
    conversion_prob = clip_probability(0.045 + 0.010 * (group == 'B') + 0.006 * (region == 'east') + 0.004 * (device == 'desktop') + rng.normal(0, 0.015, n))
    converted = np.where(rng.random(n) < conversion_prob, 1, 0)
    order_value = (rng.normal(118, 26, n) + 12 * (group == 'B') + 8 * converted).clip(18, 360)
    satisfaction = (rng.normal(7.1, 1.3, n) + 0.35 * (group == 'B')).clip(1, 10)
    return pd.DataFrame({'experiment_group': group, 'region': region, 'device_type': device, 'session_seconds': session_seconds.round(1), 'impressions': impressions, 'clicks': clicks, 'converted': converted, 'order_value': order_value.round(2), 'satisfaction_score': satisfaction.round(1)})


def build_market_research(rng):
    n = 1600
    return pd.DataFrame({
        'age_band': choice(rng, ['18_24', '25_34', '35_44', '45_54', '55_plus'], n, p=[0.17, 0.30, 0.24, 0.18, 0.11]),
        'city': choice(rng, ['beijing', 'shanghai', 'guangzhou', 'shenzhen', 'chengdu', 'wuhan'], n),
        'brand_preference': choice(rng, ['brand_a', 'brand_b', 'brand_c'], n, p=[0.38, 0.34, 0.28]),
        'channel_preference': choice(rng, ['online', 'offline', 'hybrid'], n, p=[0.36, 0.22, 0.42]),
        'price_sensitivity': rng.integers(1, 6, n),
        'nps': rng.normal(32, 21, n).clip(-60, 90).round(1),
        'brand_awareness': rng.integers(1, 6, n),
        'brand_trust': rng.integers(1, 6, n),
        'packaging_score': rng.integers(1, 6, n),
        'innovation_score': rng.integers(1, 6, n),
        'service_score': rng.integers(1, 6, n),
    })


def build_pca_sensor(rng):
    n = 2000
    latent_a = rng.normal(0, 1, n)
    latent_b = rng.normal(0, 1, n)
    latent_c = rng.normal(0, 1, n)
    return pd.DataFrame({
        'sensor_1': 0.92 * latent_a + rng.normal(0, 0.2, n),
        'sensor_2': 0.86 * latent_a + rng.normal(0, 0.25, n),
        'sensor_3': 0.81 * latent_a + rng.normal(0, 0.28, n),
        'sensor_4': 0.89 * latent_b + rng.normal(0, 0.22, n),
        'sensor_5': 0.77 * latent_b + rng.normal(0, 0.30, n),
        'sensor_6': 0.72 * latent_b + rng.normal(0, 0.32, n),
        'sensor_7': 0.93 * latent_c + rng.normal(0, 0.2, n),
        'sensor_8': 0.84 * latent_c + rng.normal(0, 0.25, n),
        'sensor_9': 0.74 * latent_c + rng.normal(0, 0.30, n),
        'sensor_10': 0.45 * latent_a + 0.32 * latent_b + rng.normal(0, 0.35, n),
    }).round(4)


def build_chi_square_store(rng):
    n = 1800
    return pd.DataFrame({
        'gender': choice(rng, ['female', 'male'], n, p=[0.54, 0.46]),
        'age_band': choice(rng, ['18_24', '25_34', '35_44', '45_plus'], n, p=[0.19, 0.36, 0.27, 0.18]),
        'membership_level': choice(rng, ['none', 'silver', 'gold'], n, p=[0.42, 0.38, 0.20]),
        'purchase_channel': choice(rng, ['app', 'store', 'web'], n, p=[0.39, 0.31, 0.30]),
        'store_format': choice(rng, ['mall', 'community', 'flagship'], n, p=[0.43, 0.34, 0.23]),
        'preferred_category': choice(rng, ['beauty', 'fashion', 'digital', 'grocery'], n, p=[0.23, 0.29, 0.18, 0.30]),
    })


def build_training_effect_ttest(rng):
    n = 1600
    training_group = choice(rng, ['control', 'trained'], n, p=[0.5, 0.5])
    department = choice(rng, ['sales', 'service', 'operations', 'it'], n)
    pre_test_score = rng.normal(68, 10, n).clip(30, 95)
    attendance_rate = rng.normal(0.86, 0.11, n).clip(0.35, 1.0)
    practice_hours = rng.normal(10, 4.6, n).clip(1, 32)
    post_test_score = (
        pre_test_score
        + 2.4
        + 6.8 * (training_group == 'trained')
        + 5.2 * attendance_rate
        + 0.38 * practice_hours
        + 1.5 * (department == 'service')
        + rng.normal(0, 4.5, n)
    ).clip(35, 100)
    return pd.DataFrame({
        'training_group': training_group,
        'department': department,
        'pre_test_score': pre_test_score.round(1),
        'attendance_rate': attendance_rate.round(3),
        'practice_hours': practice_hours.round(1),
        'post_test_score': post_test_score.round(1),
        'score_delta': (post_test_score - pre_test_score).round(1),
    })


def build_drug_response_anova(rng):
    n = 1500
    treatment_group = choice(rng, ['placebo', 'low_dose', 'mid_dose', 'high_dose'], n, p=[0.24, 0.26, 0.27, 0.23])
    center = choice(rng, ['center_a', 'center_b', 'center_c', 'center_d'], n)
    age = rng.integers(21, 79, n)
    baseline_bp = rng.normal(148, 14, n).clip(118, 198)
    adherence = rng.normal(0.78, 0.15, n).clip(0.3, 1.0)
    bp_reduction = (
        1.6
        + 4.5 * (treatment_group == 'low_dose')
        + 8.2 * (treatment_group == 'mid_dose')
        + 11.4 * (treatment_group == 'high_dose')
        + 7.5 * adherence
        - 0.045 * (baseline_bp - 148)
        - 0.02 * (age - 45)
        + 0.9 * (center == 'center_b')
        + rng.normal(0, 3.1, n)
    )
    side_effect_score = (
        2.0
        + 0.9 * (treatment_group == 'low_dose')
        + 1.7 * (treatment_group == 'mid_dose')
        + 2.6 * (treatment_group == 'high_dose')
        + rng.normal(0, 0.9, n)
    ).clip(0, 10)
    return pd.DataFrame({
        'treatment_group': treatment_group,
        'center': center,
        'age': age,
        'baseline_bp': baseline_bp.round(1),
        'adherence_rate': adherence.round(3),
        'bp_reduction': bp_reduction.round(2),
        'side_effect_score': side_effect_score.round(1),
    })


def build_ad_spend_linear_regression(rng):
    n = 1800
    tv_spend = rng.normal(26, 11, n).clip(0, 70)
    search_spend = rng.normal(15, 6, n).clip(0, 40)
    social_spend = rng.normal(9, 4, n).clip(0, 26)
    email_spend = rng.normal(4.5, 2.1, n).clip(0, 14)
    discount_rate = rng.normal(0.10, 0.05, n).clip(0, 0.32)
    season_index = rng.normal(1.0, 0.18, n).clip(0.6, 1.6)
    leads = (
        180
        + 3.8 * tv_spend
        + 5.5 * search_spend
        + 6.8 * social_spend
        + 2.9 * email_spend
        + 95 * discount_rate
        + 52 * season_index
        + rng.normal(0, 18, n)
    )
    revenue = (
        120
        + 2.4 * tv_spend
        + 4.2 * search_spend
        + 3.6 * social_spend
        + 1.5 * email_spend
        + 0.46 * leads
        + 210 * discount_rate
        + 86 * season_index
        + rng.normal(0, 28, n)
    )
    return pd.DataFrame({
        'tv_spend_k': tv_spend.round(2),
        'search_spend_k': search_spend.round(2),
        'social_spend_k': social_spend.round(2),
        'email_spend_k': email_spend.round(2),
        'discount_rate': discount_rate.round(3),
        'season_index': season_index.round(3),
        'sales_leads': leads.round(1),
        'revenue_k': revenue.round(2),
    })


def build_loan_approval_logit(rng):
    n = 2000
    income = rng.normal(20, 7.5, n).clip(3, 60)
    debt_ratio = rng.normal(0.34, 0.16, n).clip(0.02, 1.05)
    credit_score = rng.normal(645, 68, n).clip(420, 820)
    employment_years = rng.normal(6.2, 4.1, n).clip(0, 32)
    existing_loans = rng.poisson(1.4, n)
    home_ownership = choice(rng, ['rent', 'mortgage', 'own'], n, p=[0.43, 0.40, 0.17])
    marital_status = choice(rng, ['single', 'married'], n, p=[0.46, 0.54])
    application_channel = choice(rng, ['branch', 'web', 'partner'], n, p=[0.31, 0.49, 0.20])
    approval_score = (
        2.1
        + 0.055 * income
        - 2.8 * debt_ratio
        + 0.009 * (credit_score - 600)
        + 0.07 * employment_years
        - 0.19 * existing_loans
        + 0.24 * (home_ownership == 'own')
        + 0.10 * (home_ownership == 'mortgage')
        + 0.18 * (marital_status == 'married')
        - 0.16 * (application_channel == 'partner')
        + rng.normal(0, 0.55, n)
    )
    approved = np.where(rng.random(n) < sigmoid(approval_score), 'approved', 'rejected')
    return pd.DataFrame({
        'income_k': income.round(2),
        'debt_ratio': debt_ratio.round(3),
        'credit_score': credit_score.round(0),
        'employment_years': employment_years.round(1),
        'existing_loans': existing_loans,
        'home_ownership': home_ownership,
        'marital_status': marital_status,
        'application_channel': application_channel,
        'approved': approved,
    })


def build_exam_score_correlation_cn(rng):
    n = 1200
    quantitative = rng.normal(0, 1, n)
    language = rng.normal(0, 1, n)
    diligence = rng.normal(0, 1, n)
    return pd.DataFrame({
        '数学成绩': np.clip(76 + 10 * quantitative + 4 * diligence + rng.normal(0, 5, n), 35, 100).round(1),
        '统计学成绩': np.clip(74 + 9 * quantitative + 5 * diligence + rng.normal(0, 5, n), 30, 100).round(1),
        '编程成绩': np.clip(72 + 11 * quantitative + 3 * diligence + rng.normal(0, 6, n), 25, 100).round(1),
        '英语成绩': np.clip(75 + 9 * language + 2 * diligence + rng.normal(0, 5, n), 30, 100).round(1),
        '阅读理解': np.clip(77 + 10 * language + 2 * diligence + rng.normal(0, 5, n), 35, 100).round(1),
        '课堂参与度': np.clip(3.0 + 0.5 * diligence + 0.2 * language + rng.normal(0, 0.6, n), 1, 5).round(1),
        '作业完成率': np.clip(0.76 + 0.12 * diligence + rng.normal(0, 0.08, n), 0.3, 1.0).round(3),
        '睡眠时长': np.clip(7.0 - 0.15 * diligence + rng.normal(0, 0.7, n), 4.5, 9.5).round(1),
        '复习时长_小时': np.clip(10 + 2.2 * diligence + 0.8 * quantitative + rng.normal(0, 1.8, n), 1, 22).round(1),
    })


def build_dirty_customer_profile(rng):
    n = 1500
    signup_date = pd.Timestamp('2022-01-01') + pd.to_timedelta(rng.integers(0, 900, n), unit='D')
    df = pd.DataFrame({'customer_id': np.arange(100001, 100001 + n), 'signup_date': signup_date, 'city': choice(rng, ['beijing', 'shanghai', 'hangzhou', 'chengdu', 'wuhan'], n), 'age': rng.normal(35, 9, n).clip(18, 70).round(0), 'income_k': rng.normal(19, 8, n).clip(2, 65).round(2), 'purchase_count': rng.poisson(6.5, n), 'avg_order_value': rng.normal(145, 58, n).clip(12, 520).round(2), 'last_login_days': rng.integers(0, 365, n), 'customer_tag': choice(rng, ['new', 'active', 'vip', 'sleeping'], n), 'retention_label': choice(rng, ['retain', 'risk'], n, p=[0.73, 0.27])})
    df = add_missing(rng, df, ['age', 'income_k', 'avg_order_value', 'customer_tag'], 0.08)
    return add_outliers(rng, df, 'avg_order_value', multiplier=6.5, rate=0.02)


def build_dirty_property_listing(rng):
    n = 1200
    posted_date = pd.Timestamp('2023-01-01') + pd.to_timedelta(rng.integers(0, 520, n), unit='D')
    df = pd.DataFrame({'listing_id': np.arange(50001, 50001 + n), 'posted_date': posted_date, 'district': choice(rng, ['core', 'inner_ring', 'suburb'], n, p=[0.25, 0.43, 0.32]), 'area_sqm': rng.normal(102, 34, n).clip(28, 260).round(1), 'bedrooms': rng.integers(1, 6, n), 'building_age': rng.integers(0, 38, n), 'floor': rng.integers(1, 33, n), 'near_subway': choice(rng, ['yes', 'no'], n, p=[0.63, 0.37]), 'renovation_level': choice(rng, ['simple', 'standard', 'premium'], n), 'price_k': rng.normal(245, 98, n).clip(35, 980).round(2)})
    df = add_missing(rng, df, ['area_sqm', 'building_age', 'renovation_level'], 0.06)
    return add_outliers(rng, df, 'price_k', multiplier=4.8, rate=0.018)


def build_dirty_sales_pipeline(rng):
    n = 1300
    created_date = pd.Timestamp('2023-06-01') + pd.to_timedelta(rng.integers(0, 420, n), unit='D')
    df = pd.DataFrame({'opportunity_id': np.arange(80001, 80001 + n), 'created_date': created_date, 'channel': choice(rng, ['partner', 'inbound', 'outbound', 'event'], n), 'industry': choice(rng, ['retail', 'manufacturing', 'education', 'healthcare', 'it'], n), 'lead_score': rng.normal(63, 18, n).clip(3, 100).round(1), 'deal_size_k': rng.normal(38, 22, n).clip(1, 180).round(2), 'sales_cycle_days': rng.normal(42, 19, n).clip(3, 160).round(0), 'touchpoints': rng.poisson(6.2, n) + 1, 'owner_region': choice(rng, ['east', 'north', 'south', 'west'], n), 'status': choice(rng, ['won', 'open', 'lost'], n, p=[0.29, 0.36, 0.35])})
    df = add_missing(rng, df, ['lead_score', 'deal_size_k', 'industry'], 0.07)
    return add_outliers(rng, df, 'sales_cycle_days', multiplier=5.0, rate=0.02)


def build_dirty_order_master(rng):
    n = 1350
    order_date = pd.Timestamp('2024-01-01') + pd.to_timedelta(rng.integers(0, 420, n), unit='D')
    date_format = choice(rng, ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y'], n, p=[0.48, 0.32, 0.20])
    order_date_text = [item.strftime(fmt) for item, fmt in zip(order_date, date_format)]
    customer_level = choice(rng, ['VIP', 'vip', 'Vip ', '普通', ' normal'], n, p=[0.18, 0.16, 0.10, 0.40, 0.16])
    order_amount = rng.normal(860, 420, n).clip(25, 5200).round(2).astype(object)
    unit_price = rng.normal(126, 48, n).clip(8, 980).round(2).astype(object)
    quantity = (rng.poisson(3.6, n) + 1).astype(object)

    amount_mask = rng.random(n) < 0.08
    unit_mask = rng.random(n) < 0.07
    quantity_mask = rng.random(n) < 0.06
    order_amount[amount_mask] = [f'{value:,.2f}' for value in rng.normal(1380, 520, amount_mask.sum()).clip(100, 6800)]
    unit_price[unit_mask] = [f'{value:,.0f}' for value in rng.normal(168, 60, unit_mask.sum()).clip(10, 1500)]
    quantity[quantity_mask] = [f'{value}件' for value in rng.integers(1, 9, quantity_mask.sum())]

    df = pd.DataFrame({
        'order_id': np.arange(700001, 700001 + n),
        'order_date': order_date_text,
        'city': choice(rng, ['beijing', 'Beijing', 'shanghai', 'SH', 'chengdu', 'wuhan '], n),
        'channel_code': choice(rng, ['APP', 'app', 'web', 'WEB ', 'store', 'Store'], n),
        'sku_code': choice(rng, ['SKU-1001', 'SKU-1002', 'sku-1003', 'SKU1004', 'sku_1005'], n),
        'quantity': quantity,
        'unit_price': unit_price,
        'order_amount': order_amount,
        'customer_level': customer_level,
        'status': choice(rng, ['paid', 'shipped', 'refund', 'closed'], n, p=[0.42, 0.29, 0.11, 0.18]),
    })
    df = add_missing(rng, df, ['city', 'channel_code', 'order_amount', 'customer_level'], 0.06)
    duplicate_rows = df.sample(n=55, random_state=20260416).copy()
    duplicate_rows['order_id'] = duplicate_rows['order_id'].values
    dirty_df = pd.concat([df, duplicate_rows], ignore_index=True)
    if 'order_amount' in dirty_df.columns:
        outlier_mask = rng.random(len(dirty_df)) < 0.015
        dirty_df.loc[outlier_mask, 'order_amount'] = '99999.99'
    return dirty_df


def build_dirty_supplier_quality(rng):
    n = 1250
    inspection_date = pd.Timestamp('2024-02-01') + pd.to_timedelta(rng.integers(0, 360, n), unit='D')
    inspection_style = choice(rng, ['%Y-%m-%d', '%Y/%m/%d'], n, p=[0.62, 0.38])
    inspection_text = [item.strftime(fmt) for item, fmt in zip(inspection_date, inspection_style)]
    pass_rate = rng.normal(92, 6.5, n).clip(55, 100).round(1).astype(object)
    delay_days = rng.normal(4.8, 4.2, n).clip(-2, 35).round(1).astype(object)
    pass_rate_mask = rng.random(n) < 0.09
    delay_mask = rng.random(n) < 0.07
    pass_rate[pass_rate_mask] = [f'{value}%' for value in rng.integers(68, 101, pass_rate_mask.sum())]
    delay_days[delay_mask] = choice(rng, ['N/A', 'unknown', '待补录'], delay_mask.sum())
    df = pd.DataFrame({
        'supplier_id': np.arange(90001, 90001 + n),
        'inspection_date': inspection_text,
        'supplier_type': choice(rng, ['A', 'a', 'A级', 'B', 'B ', 'C'], n, p=[0.19, 0.12, 0.12, 0.26, 0.14, 0.17]),
        'batch_pass_rate': pass_rate,
        'delay_days': delay_days,
        'complaint_count': rng.poisson(1.4, n),
        'material_score': rng.normal(78, 12, n).clip(20, 100).round(1),
        'region': choice(rng, ['east', 'north', 'south', 'west', '华东', '华南'], n),
        'inspector': choice(rng, ['Zhang', 'Li', 'Wang', 'Chen', 'Sun'], n),
        'risk_flag': choice(rng, ['low', 'medium', 'high'], n, p=[0.56, 0.28, 0.16]),
    })
    df = add_missing(rng, df, ['supplier_type', 'batch_pass_rate', 'region'], 0.05)
    df = add_outliers(rng, df, 'material_score', multiplier=1.8, rate=0.018)
    duplicate_rows = df.sample(n=45, random_state=20260417).copy()
    return pd.concat([df, duplicate_rows], ignore_index=True)


DATASETS = [
    DatasetSpec('classification', 'customer_churn_classification.csv', '分类', '电信客户流失', '分类建模 / 编码 / 缺失值 / 混淆矩阵', build_customer_churn),
    DatasetSpec('classification', 'credit_risk_classification.csv', '分类', '信贷违约风险', '二分类 / 逻辑回归 / 随机森林', build_credit_risk),
    DatasetSpec('classification', 'equipment_fault_multiclass.csv', '分类', '设备健康状态多分类', '多分类 / 工业预警 / 混淆矩阵', build_equipment_fault_multiclass),
    DatasetSpec('classification', 'hr_attrition_classification.csv', '分类', '员工离职预测', '分类建模 / 特征解释', build_hr_attrition),
    DatasetSpec('classification', 'insurance_claim_fraud_imbalanced.csv', '分类', '保险欺诈识别', '不平衡分类 / 精确率召回率 / 风险识别', build_insurance_claim_fraud),
    DatasetSpec('classification', 'medical_screening_classification.csv', '分类', '体检筛查阳性预测', '分类建模 / 指标可视化', build_medical_screening),
    DatasetSpec('classification', 'marketing_response_classification.csv', '分类', '营销活动响应', '分类建模 / 分组分析', build_marketing_response),
    DatasetSpec('classification', 'student_dropout_classification.csv', '分类', '学生流失预测', '分类建模 / 统计分析', build_student_dropout),
    DatasetSpec('classification', 'subscription_upgrade_cn.csv', '分类', 'SaaS 套餐升级建议', '中文字段 / 多分类 / 业务演示', build_app_subscription_upgrade_cn),
    DatasetSpec('quick_demo', 'quick_classification_demo.csv', '快速演示', '快速分类演示', '小样本 / 上手演示 / UI 联调', sample_builder(build_customer_churn, 180)),
    DatasetSpec('regression', 'housing_price_regression.csv', '回归', '房价预测', '回归建模 / 异常值处理', build_housing_price),
    DatasetSpec('regression', 'sales_forecast_regression.csv', '回归', '销量预测', '回归建模 / 时间字段演示', build_sales_forecast),
    DatasetSpec('regression', 'manufacturing_quality_regression.csv', '回归', '制造质量评分', '回归建模 / 工业场景', build_manufacturing_quality),
    DatasetSpec('regression', 'medical_cost_regression.csv', '回归', '年度医疗费用预测', '回归建模 / 医疗业务演示', build_medical_cost_regression),
    DatasetSpec('regression', 'energy_consumption_regression.csv', '回归', '能耗预测', '回归建模 / 多类型字段', build_energy_consumption),
    DatasetSpec('regression', 'ecommerce_order_value_regression.csv', '回归', '电商订单金额预测', '回归建模 / 分类+数值混合', build_ecommerce_order_value),
    DatasetSpec('regression', 'logistics_delivery_time_regression.csv', '回归', '物流时效预测', '回归建模 / 业务演示', build_logistics_delivery_time),
    DatasetSpec('regression', 'store_revenue_regression_cn.csv', '回归', '门店营业额预测', '中文字段 / 回归建模 / 线下零售', build_store_revenue_regression_cn),
    DatasetSpec('quick_demo', 'quick_regression_demo.csv', '快速演示', '快速回归演示', '小样本 / 回归联调 / 快速出图', sample_builder(build_housing_price, 200)),
    DatasetSpec('statistics', 'ad_spend_linear_regression.csv', '统计', '广告投放线性回归', '线性回归 / 相关分析 / 回归系数解释', build_ad_spend_linear_regression),
    DatasetSpec('statistics', 'retail_segmentation_clustering.csv', '统计', '零售客群聚类', '聚类 / PCA / 描述统计', build_retail_segmentation),
    DatasetSpec('statistics', 'employee_satisfaction_factor_analysis.xlsx', '统计', '员工满意度因子分析', '因子分析 / 描述统计 / Excel 上传', build_employee_factor),
    DatasetSpec('statistics', 'ab_test_experiment.csv', '统计', 'A/B 实验分析', 'T 检验 / 方差分析 / 描述统计', build_ab_test),
    DatasetSpec('statistics', 'drug_response_anova.csv', '统计', '药物组间差异分析', '方差分析 / 多组比较 / 医疗研究', build_drug_response_anova),
    DatasetSpec('statistics', 'exam_score_correlation_cn.xlsx', '统计', '学生成绩相关分析', '中文字段 / 相关分析 / PCA', build_exam_score_correlation_cn),
    DatasetSpec('statistics', 'loan_approval_logit.csv', '统计', '贷款审批 Logit 回归', 'Logit 回归 / 二分类解释', build_loan_approval_logit),
    DatasetSpec('statistics', 'market_research_survey.csv', '统计', '市场调研问卷', '卡方 / 描述统计 / 相关分析', build_market_research),
    DatasetSpec('statistics', 'pca_sensor_monitoring.csv', '统计', '传感器主成分分析', 'PCA / 相关分析', build_pca_sensor),
    DatasetSpec('statistics', 'chi_square_store_preference.csv', '统计', '渠道偏好卡方检验', '卡方 / 交叉表', build_chi_square_store),
    DatasetSpec('statistics', 'training_effect_t_test.csv', '统计', '培训效果 T 检验', '独立样本 T 检验 / 前后差值分析', build_training_effect_ttest),
    DatasetSpec('quick_demo', 'quick_statistics_demo.xlsx', '快速演示', '快速统计演示', '小样本 / 描述统计 / Excel 上传', sample_builder(build_ab_test, 160)),
    DatasetSpec('dirty_data', 'dirty_customer_profile.csv', '脏数据', '客户画像脏数据', '缺失值 / 异常值 / 日期字段 / 编码', build_dirty_customer_profile),
    DatasetSpec('dirty_data', 'dirty_order_master.csv', '脏数据', '订单主数据脏数据', '重复值 / 混合编码 / 日期清洗 / 类型识别', build_dirty_order_master),
    DatasetSpec('dirty_data', 'dirty_property_listing.xlsx', '脏数据', '房源信息脏数据', 'Excel 上传 / 缺失值 / 异常值', build_dirty_property_listing),
    DatasetSpec('dirty_data', 'dirty_sales_pipeline.csv', '脏数据', '销售漏斗脏数据', '缺失值 / 异常值 / 筛选 / 派生变量', build_dirty_sales_pipeline),
    DatasetSpec('dirty_data', 'dirty_supplier_quality.xlsx', '脏数据', '供应商质检脏数据', '重复值 / 缺失值 / 百分比文本 / Excel 演示', build_dirty_supplier_quality),
    DatasetSpec('quick_demo', 'quick_dirty_demo.csv', '快速演示', '快速脏数据演示', '小样本 / 清洗联调 / 滚动预览', sample_builder(build_dirty_sales_pipeline, 180)),
]


def write_dataset(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == '.xlsx':
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')


def build_readme(records):
    lines = [
        '# 演示数据集库',
        '',
        '这批数据集用于测试演示，覆盖分类、回归、统计分析、聚类/PCA、脏数据预处理等场景。',
        '',
        '## 目录结构',
        '',
        '- `classification/` 分类建模示例',
        '- `regression/` 回归建模示例',
        '- `statistics/` 统计分析与聚类/PCA 示例',
        '- `dirty_data/` 缺失值、异常值、日期字段、混合类型示例',
        '- `quick_demo/` 小样本快速演示数据',
        '',
        '## 数据集清单',
        '',
        '| 文件 | 模块 | 场景 | 行数 | 列数 | 推荐用途 |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    for record in records:
        lines.append(f"| `{record['relative_path']}` | {record['module']} | {record['scenario']} | {record['rows']} | {record['columns']} | {record['recommended_usage']} |")
    return '\n'.join(lines) + '\n'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for index, spec in enumerate(DATASETS, start=1):
        rng = np.random.default_rng(20260416 + index * 97)
        df = spec.builder(rng)
        output_path = OUTPUT_DIR / spec.folder / spec.filename
        write_dataset(df, output_path)
        records.append({'module': spec.module, 'scenario': spec.scenario, 'recommended_usage': spec.recommended_usage, 'filename': spec.filename, 'relative_path': str(output_path.relative_to(OUTPUT_DIR)).replace('\\', '/'), 'rows': len(df), 'columns': len(df.columns), 'format': output_path.suffix.lower().lstrip('.')})
    catalog = pd.DataFrame(records).sort_values(['module', 'scenario', 'filename']).reset_index(drop=True)
    catalog.to_csv(OUTPUT_DIR / 'dataset_catalog.csv', index=False, encoding='utf-8-sig')
    (OUTPUT_DIR / 'README.md').write_text(build_readme(catalog.to_dict(orient='records')), encoding='utf-8')
    print(f'Generated {len(records)} demo datasets in: {OUTPUT_DIR}')
    for item in catalog.to_dict(orient='records'):
        print(f"- {item['relative_path']} ({item['rows']} rows)")


if __name__ == '__main__':
    main()
