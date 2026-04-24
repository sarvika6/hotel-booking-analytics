"""
Hotel Booking Intelligence — Full Analysis Script
Run this script to generate all charts and the clean CSV.
Equivalent to running the full Jupyter notebook front-to-back.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings, os, sys

warnings.filterwarnings('ignore')

# ── Global plot defaults ───────────────────────────────────────────────────
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
sns.set_theme(style='whitegrid')
sns.set_palette('husl')

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, 'data',    'hotel_bookings.csv')
EXPORT_DIR = os.path.join(BASE, 'exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

print("✅ All libraries loaded successfully")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — LOADING DATA")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"📐 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — DATA QUALITY AUDIT")
print("="*60)

null_counts = df.isnull().sum()
null_cols   = null_counts[null_counts > 0]
null_pct    = (null_cols / len(df) * 100).round(2)
null_df     = pd.DataFrame({'Null Count': null_cols, 'Null %': null_pct})
print("\n1️⃣  NULL VALUES:")
print(null_df.to_string())

dupe_count = df.duplicated().sum()
print(f"\n3️⃣  DUPLICATES: {dupe_count:,} duplicate rows found")

range_cols = ['adr', 'lead_time', 'stays_in_week_nights',
              'stays_in_weekend_nights', 'adults', 'children']
print("\n4️⃣  VALUE RANGES:")
print(df[range_cols].describe().round(2).to_string())
print(f"   - Negative ADR rows : {(df['adr'] < 0).sum()}")
print(f"   - ADR > 5000 rows   : {(df['adr'] > 5000).sum()}")

cat_cols = ['hotel','market_segment','distribution_channel','meal','reservation_status']
print("\n5️⃣  CATEGORICAL DISTRIBUTIONS:")
for col in cat_cols:
    print(f"\n  [{col}] — {df[col].nunique()} unique values:")
    print(df[col].value_counts().to_string())

print("\n✅ Data Quality Audit Complete")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — CLEANING & FEATURE ENGINEERING")
print("="*60)

original_shape = df.shape

# ── Null fixes ────────────────────────────────────────────────────────────
df['children'] = df['children'].fillna(0)
df['country']  = df['country'].fillna('Unknown')
df['agent']    = df['agent'].fillna(0)
df['company']  = df['company'].fillna(0)

# ── Type fixes ────────────────────────────────────────────────────────────
df['children'] = df['children'].astype(int)
df['agent']    = df['agent'].astype(int)
df['company']  = df['company'].astype(int)

# ── Invalid row removal ───────────────────────────────────────────────────
before = len(df)
df = df[~((df['adults'] + df['children'] + df['babies']) == 0)]
df = df[~((df['stays_in_week_nights'] + df['stays_in_weekend_nights']) == 0)]
df = df[df['adr'] >= 0]
df = df[df['adr'] <= 5000]
print(f"Rows removed (invalid): {before - len(df):,}")

# ── Feature engineering ───────────────────────────────────────────────────
df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['revenue']      = df['adr'] * df['total_nights']
df['is_family']    = ((df['children'] + df['babies']) > 0).astype(int)

MONTH_MAP = {
    'January':1,'February':2,'March':3,'April':4,
    'May':5,'June':6,'July':7,'August':8,
    'September':9,'October':10,'November':11,'December':12
}
df['month_num'] = df['arrival_date_month'].map(MONTH_MAP)

try:
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['month_num'].astype(str).str.zfill(2) + '-' +
        df['arrival_date_day_of_month'].astype(str).str.zfill(2),
        errors='coerce'
    )
    print(f"arrival_date range: {df['arrival_date'].min()} → {df['arrival_date'].max()}")
except Exception as e:
    print(f"⚠️ Date parse issue: {e}")
    df['arrival_date'] = pd.NaT

lead_bins   = [0, 30, 90, 180, 365, df['lead_time'].max() + 1]
lead_labels = ['0–30 days','31–90 days','91–180 days','181–365 days','365+ days']
df['lead_time_bucket'] = pd.cut(df['lead_time'], bins=lead_bins,
                                 labels=lead_labels, right=True)

def classify_stay(row):
    wk, we = row['stays_in_week_nights'], row['stays_in_weekend_nights']
    if wk > 0 and we == 0: return 'Weekday Only'
    if we > 0 and wk == 0: return 'Weekend Only'
    return 'Mixed'

df['stay_type'] = df.apply(classify_stay, axis=1)

df['revenue_tier'] = pd.qcut(df['revenue'], q=4,
                              labels=['Low','Medium','High','Premium'],
                              duplicates='drop')

df['repeat_guest_label'] = df['is_repeated_guest'].map(
    {1:'Repeat Guest', 0:'New Guest'})

print(f"\nOriginal shape : {original_shape[0]:,} × {original_shape[1]}")
print(f"Final shape    : {len(df):,} × {len(df.columns)}")
new_cols = ['total_nights','total_guests','revenue','is_family','arrival_date',
            'month_num','lead_time_bucket','stay_type','revenue_tier','repeat_guest_label']
for c in new_cols:
    print(f"  ✔ {c}")
print("\n✅ Data Cleaning & Feature Engineering Complete")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — MONTHLY CANCELLATION RATES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 1 — MONTHLY CANCELLATION RATES")
print("="*60)

MONTH_ORDER  = ['January','February','March','April','May','June',
                'July','August','September','October','November','December']
SHORT_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

monthly = (df.groupby('arrival_date_month')['is_canceled']
             .agg(['mean','count']).reset_index())
monthly.columns = ['month','cancel_rate','total_bookings']
monthly['cancel_rate'] *= 100
monthly['month_num'] = monthly['month'].map(
    {m: i+1 for i,m in enumerate(MONTH_ORDER)})
monthly = monthly.sort_values('month_num').reset_index(drop=True)
monthly['short_month'] = [SHORT_MONTHS[int(n)-1] for n in monthly['month_num']]

hotel_monthly = (df.groupby(['arrival_date_month','hotel'])['is_canceled']
                   .mean().reset_index())
hotel_monthly.columns = ['month','hotel','cancel_rate']
hotel_monthly['cancel_rate'] *= 100
hotel_monthly['month_num'] = hotel_monthly['month'].map(
    {m: i+1 for i,m in enumerate(MONTH_ORDER)})
hotel_monthly = hotel_monthly.sort_values(['hotel','month_num']).reset_index(drop=True)
hotel_monthly['short_month'] = [SHORT_MONTHS[int(n)-1] for n in hotel_monthly['month_num']]

avg_rate   = monthly['cancel_rate'].mean()
peak_month = monthly.loc[monthly['cancel_rate'].idxmax(), 'short_month']
peak_rate  = monthly['cancel_rate'].max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Business Question 1 — Monthly Cancellation Rate Analysis',
             fontsize=14, fontweight='bold')

bar_colors = ['crimson' if r == monthly['cancel_rate'].max() else 'steelblue'
              for r in monthly['cancel_rate']]
bars = ax1.bar(monthly['short_month'], monthly['cancel_rate'],
               color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
ax1.axhline(avg_rate, color='darkorange', linestyle='--', linewidth=2,
            label=f'Average: {avg_rate:.1f}%', zorder=4)
for bar, val in zip(bars, monthly['cancel_rate']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax1.set_title('Monthly Cancellation Rate — All Hotels', fontweight='bold')
ax1.set_xlabel('Month'); ax1.set_ylabel('Cancellation Rate (%)')
ax1.legend(); ax1.set_ylim(0, monthly['cancel_rate'].max() * 1.18)
ax1.grid(axis='y', alpha=0.4)

city_d   = hotel_monthly[hotel_monthly['hotel']=='City Hotel'].reset_index(drop=True)
resort_d = hotel_monthly[hotel_monthly['hotel']=='Resort Hotel'].reset_index(drop=True)
x     = np.arange(len(SHORT_MONTHS))
width = 0.38

# Align both series to all 12 months
city_rates   = []
resort_rates = []
for sm in SHORT_MONTHS:
    c = city_d[city_d['short_month']==sm]['cancel_rate']
    r = resort_d[resort_d['short_month']==sm]['cancel_rate']
    city_rates.append(float(c.iloc[0]) if len(c) else 0)
    resort_rates.append(float(r.iloc[0]) if len(r) else 0)

ax2.bar(x - width/2, city_rates,   width, label='City Hotel',
        color='teal',  edgecolor='white', linewidth=0.8, zorder=3)
ax2.bar(x + width/2, resort_rates, width, label='Resort Hotel',
        color='coral', edgecolor='white', linewidth=0.8, zorder=3)
ax2.set_xticks(x); ax2.set_xticklabels(SHORT_MONTHS)
ax2.set_title('Monthly Cancellation: City vs Resort Hotel', fontweight='bold')
ax2.set_xlabel('Month'); ax2.set_ylabel('Cancellation Rate (%)')
ax2.legend(); ax2.grid(axis='y', alpha=0.4)

plt.tight_layout()
p1 = os.path.join(EXPORT_DIR, 'chart1_cancellation_by_month.png')
plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p1}")
print(f"   Peak: {peak_month} at {peak_rate:.1f}% | Avg: {avg_rate:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — REVENUE BY MARKET SEGMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 2 — REVENUE BY MARKET SEGMENT")
print("="*60)

df_conf = df[df['is_canceled'] == 0].copy()

seg_rev = (df_conf.groupby('market_segment')
           .agg(total_revenue=('revenue','sum'),
                avg_adr=('adr','mean'),
                total_bookings=('adr','count'))
           .reset_index()
           .sort_values('total_revenue', ascending=False))

n = len(seg_rev)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Business Question 2 — Revenue by Market Segment / Booking Channel',
             fontsize=14, fontweight='bold')

colors_teal  = plt.cm.GnBu(np.linspace(0.9, 0.4, n))
colors_coral = plt.cm.OrRd(np.linspace(0.9, 0.4, n))

bars1 = ax1.barh(seg_rev['market_segment'], seg_rev['total_revenue']/1e6,
                 color=colors_teal, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars1, seg_rev['total_revenue']):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'€{val/1e6:.1f}M', va='center', fontsize=9, fontweight='bold')
ax1.set_title('Total Revenue by Booking Channel', fontweight='bold')
ax1.set_xlabel('Revenue (€ Millions)'); ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.4)

seg_adr = seg_rev.sort_values('avg_adr', ascending=False)
bars2 = ax2.barh(seg_adr['market_segment'], seg_adr['avg_adr'],
                 color=colors_coral, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars2, seg_adr['avg_adr']):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'€{val:.0f}', va='center', fontsize=9, fontweight='bold')
ax2.set_title('Average Daily Rate by Channel', fontweight='bold')
ax2.set_xlabel('ADR (€ per night)'); ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.4)

ax3.scatter(seg_rev['total_bookings'], seg_rev['total_revenue']/1e6,
            s=seg_rev['avg_adr']*3, color='mediumpurple',
            alpha=0.8, edgecolors='purple', linewidth=1.5, zorder=5)
for _, row in seg_rev.iterrows():
    ax3.annotate(row['market_segment'],
                 (row['total_bookings'], row['total_revenue']/1e6),
                 fontsize=8, ha='center', va='bottom',
                 xytext=(0, 8), textcoords='offset points')
ax3.set_title('Volume vs Revenue by Segment', fontweight='bold')
ax3.set_xlabel('Total Bookings'); ax3.set_ylabel('Total Revenue (€M)')
ax3.grid(alpha=0.4)

plt.tight_layout()
p2 = os.path.join(EXPORT_DIR, 'chart2_revenue_by_segment.png')
plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p2}")
top_seg = seg_rev.iloc[0]
print(f"   Top segment: {top_seg['market_segment']} — €{top_seg['total_revenue']/1e6:.1f}M")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — LEAD TIME CANCELLATION RISK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 3 — LEAD TIME & CANCELLATION RISK")
print("="*60)

lead_stats = (df.groupby('lead_time_bucket', observed=True)
              .agg(cancel_rate=('is_canceled','mean'),
                   total=('is_canceled','count'))
              .reset_index())
lead_stats['cancel_rate'] *= 100

# Revenue at risk per bucket
rev_risk = []
for bucket in lead_stats['lead_time_bucket']:
    mask = (df['lead_time_bucket'] == bucket) & (df['is_canceled'] == 1)
    rev_risk.append(df.loc[mask, 'revenue'].sum())
lead_stats['revenue_at_risk'] = rev_risk

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Business Question 3 — Lead Time, Cancellation Risk & Revenue at Risk',
             fontsize=14, fontweight='bold')

x_labels = lead_stats['lead_time_bucket'].astype(str)
ax1.plot(x_labels, lead_stats['cancel_rate'],
         color='darkorange', linewidth=2.5, marker='o', markersize=10, zorder=5)
ax1.fill_between(x_labels, lead_stats['cancel_rate'], alpha=0.15, color='darkorange')
max_cr = lead_stats['cancel_rate'].max()
ax1.axhspan(50, max_cr * 1.15, alpha=0.08, color='crimson', label='Danger Zone (>50%)')
ax1.axhline(50, color='crimson', linestyle='--', linewidth=1.5, alpha=0.6)
for i, (x, y) in enumerate(zip(x_labels, lead_stats['cancel_rate'])):
    ax1.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=10,
                 fontweight='bold', color='darkorange')
ax1.set_title('Cancellation Risk by Advance Booking Window', fontweight='bold')
ax1.set_xlabel('Lead Time Bucket'); ax1.set_ylabel('Cancellation Rate (%)')
ax1.legend(); ax1.set_ylim(0, max_cr * 1.3)
ax1.grid(alpha=0.4)

bars3 = ax2.bar(x_labels, lead_stats['revenue_at_risk']/1e3,
                color='crimson', edgecolor='white', linewidth=0.8, zorder=3)
max_rev = (lead_stats['revenue_at_risk']/1e3).max()
for bar, val in zip(bars3, lead_stats['revenue_at_risk']):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max_rev * 0.01,
             f'€{val/1e3:.0f}K', ha='center', va='bottom',
             fontsize=9, fontweight='bold')
ax2.set_title('Revenue at Risk from Cancellations (€)', fontweight='bold')
ax2.set_xlabel('Lead Time Bucket'); ax2.set_ylabel('Revenue at Risk (€ Thousands)')
ax2.grid(axis='y', alpha=0.4)

plt.tight_layout()
p3 = os.path.join(EXPORT_DIR, 'chart3_leadtime_cancellation.png')
plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p3}")
peak_bucket = lead_stats.loc[lead_stats['cancel_rate'].idxmax(), 'lead_time_bucket']
peak_rev    = lead_stats.loc[lead_stats['revenue_at_risk'].idxmax(), 'revenue_at_risk']
print(f"   Highest-risk window  : {peak_bucket}")
print(f"   Max revenue at risk  : €{peak_rev:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 4 — COUNTRY REVENUE ANALYSIS")
print("="*60)

df_conf2 = df[df['is_canceled'] == 0].copy()
country_stats = (df_conf2.groupby('country')
                 .agg(total_revenue=('revenue','sum'),
                      total_bookings=('adr','count'),
                      avg_adr=('adr','mean'))
                 .reset_index()
                 .sort_values('total_revenue', ascending=False))

top15 = country_stats.head(15).copy()
top20 = country_stats.head(20).copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Business Question 4 — Geographic Revenue & Guest Value Analysis',
             fontsize=14, fontweight='bold')

bar_colors_left = []
for i in range(len(top15)):
    if i == 0:   bar_colors_left.append('#FFD700')
    elif i == 1: bar_colors_left.append('#C0C0C0')
    elif i == 2: bar_colors_left.append('#CD7F32')
    else:        bar_colors_left.append('steelblue')

bars4 = ax1.barh(top15['country'][::-1],
                 top15['total_revenue'][::-1]/1e6,
                 color=bar_colors_left[::-1], edgecolor='white', linewidth=0.8)
max_rev_label = (top15['total_revenue']/1e6).max()
for bar, val in zip(bars4, top15['total_revenue'][::-1]):
    ax1.text(bar.get_width() + max_rev_label*0.01,
             bar.get_y() + bar.get_height()/2,
             f'€{val/1e6:.2f}M', va='center', fontsize=9, fontweight='bold')
ax1.set_title('Top 15 Countries by Revenue Generated', fontweight='bold')
ax1.set_xlabel('Total Revenue (€ Millions)')
ax1.grid(axis='x', alpha=0.4)

med_bk  = top20['total_bookings'].median()
med_adr = top20['avg_adr'].median()
sizes   = (top20['total_revenue'] / top20['total_revenue'].max()) * 800 + 50
sc = ax2.scatter(top20['total_bookings'], top20['avg_adr'],
                 s=sizes, c=top20['total_revenue'],
                 cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)
plt.colorbar(sc, ax=ax2, label='Total Revenue (€)')
for _, row in top20.iterrows():
    ax2.annotate(row['country'], (row['total_bookings'], row['avg_adr']),
                 fontsize=7.5, ha='center', va='bottom',
                 xytext=(0, 5), textcoords='offset points')
ax2.axvline(med_bk,  color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.axhline(med_adr, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
max_x = top20['total_bookings'].max()
ax2.text(max_x*0.98, med_adr+1,  'High Volume/High Value',   ha='right', fontsize=8, color='green',     fontweight='bold')
ax2.text(max_x*0.98, med_adr-5,  'High Volume/Low Value',    ha='right', fontsize=8, color='red',       fontweight='bold')
ax2.text(med_bk*0.02, med_adr+1, 'Low Volume/High Value',    ha='left',  fontsize=8, color='darkorange',fontweight='bold')
ax2.text(med_bk*0.02, med_adr-5, 'Niche',                    ha='left',  fontsize=8, color='gray',      fontweight='bold')
ax2.set_title('Country Segmentation: Volume vs Value', fontweight='bold')
ax2.set_xlabel('Total Bookings'); ax2.set_ylabel('Average Daily Rate (€)')
ax2.grid(alpha=0.4)

plt.tight_layout()
p4 = os.path.join(EXPORT_DIR, 'chart4_country_analysis.png')
plt.savefig(p4, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p4}")
print("   Top 3 revenue countries:")
for i, row in top15.head(3).iterrows():
    print(f"     {row['country']}: €{row['total_revenue']/1e6:.2f}M | ADR €{row['avg_adr']:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — GUEST PROFILE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 5 — GUEST PROFILE & REVENUE OPTIMISATION")
print("="*60)

df_c3 = df[df['is_canceled'] == 0].copy()

heatmap_data = df_c3.pivot_table(
    values='adr', index='hotel', columns='meal', aggfunc='mean')

stay_guest = (df_c3.groupby(['stay_type','is_family'])['revenue']
              .mean().reset_index())
stay_guest['guest_label'] = stay_guest['is_family'].map({1:'Family',0:'Non-Family'})

tier_counts = df_c3['revenue_tier'].value_counts().reindex(
    ['Low','Medium','High','Premium']).dropna()
tier_labels = list(tier_counts.index)
tier_vals   = list(tier_counts.values)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Business Question 5 — Guest Profile & Booking Revenue Optimisation',
             fontsize=14, fontweight='bold')

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', ax=ax1,
            annot_kws={'fontsize':11,'fontweight':'bold'})
ax1.set_title('Avg Daily Rate: Hotel × Meal Package', fontweight='bold')
ax1.set_xlabel('Meal Package'); ax1.set_ylabel('Hotel Type')

stay_types = ['Weekday Only','Weekend Only','Mixed']
family_vals = []
nonfam_vals = []
for st in stay_types:
    fv = stay_guest[(stay_guest['stay_type']==st)&(stay_guest['is_family']==1)]['revenue']
    nv = stay_guest[(stay_guest['stay_type']==st)&(stay_guest['is_family']==0)]['revenue']
    family_vals.append(float(fv.iloc[0]) if len(fv) else 0)
    nonfam_vals.append(float(nv.iloc[0]) if len(nv) else 0)

x2 = np.arange(len(stay_types)); w2 = 0.38
ax2.bar(x2 - w2/2, family_vals, w2, label='Family',     color='teal',  edgecolor='white')
ax2.bar(x2 + w2/2, nonfam_vals, w2, label='Non-Family', color='coral', edgecolor='white')
ax2.set_xticks(x2); ax2.set_xticklabels(stay_types)
ax2.set_title('Revenue by Stay Pattern and Guest Type', fontweight='bold')
ax2.set_xlabel('Stay Type'); ax2.set_ylabel('Average Revenue per Booking (€)')
ax2.legend(); ax2.grid(axis='y', alpha=0.4)

tier_palette = ['#AED6F1','#5DADE2','#2471A3','#154360']
explode = [0.02] * len(tier_labels)
if 'Premium' in tier_labels:
    explode[tier_labels.index('Premium')] = 0.12
wedges, texts, autotexts = ax3.pie(
    tier_vals, labels=tier_labels, autopct='%1.1f%%',
    colors=tier_palette[:len(tier_labels)], explode=explode,
    startangle=140, pctdistance=0.8)
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight('bold')
ax3.set_title('Booking Distribution by Revenue Tier', fontweight='bold')

plt.tight_layout()
p5 = os.path.join(EXPORT_DIR, 'chart5_guest_profile.png')
plt.savefig(p5, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p5}")
prem_pct = tier_counts.get('Premium',0)/tier_counts.sum()*100
print(f"   Premium tier: {prem_pct:.1f}%")
best_combo = heatmap_data.stack().idxmax()
print(f"   Best ADR combo: {best_combo[0]} + {best_combo[1]} — €{heatmap_data.stack().max():.0f}/night")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — REPEAT vs NEW GUESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHART 6 — REPEAT vs NEW GUEST ANALYSIS")
print("="*60)

repeat_stats = (df.groupby('repeat_guest_label')
                .agg(avg_adr=('adr','mean'),
                     avg_nights=('total_nights','mean'),
                     avg_lead_time=('lead_time','mean'),
                     cancel_rate=('is_canceled','mean'),
                     avg_revenue=('revenue','mean'),
                     count=('adr','count'))
                .reset_index())
repeat_stats['cancel_rate'] *= 100
repeat_stats['pct_of_total'] = repeat_stats['count'] / repeat_stats['count'].sum() * 100

print("\n  REPEAT vs NEW GUEST — COMPARISON TABLE")
print(repeat_stats[['repeat_guest_label','avg_adr','avg_nights',
                     'avg_lead_time','cancel_rate','avg_revenue',
                     'count','pct_of_total']].to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Bonus Analysis — Repeat Guest vs New Guest Comparison',
             fontsize=14, fontweight='bold')

metrics       = ['avg_adr','avg_revenue','avg_nights','avg_lead_time']
metric_labels = ['Avg ADR (€)','Avg Revenue (€)','Avg Nights','Avg Lead Time (days)']
x3 = np.arange(len(metrics)); w3 = 0.38

new_row    = repeat_stats[repeat_stats['repeat_guest_label']=='New Guest']
repeat_row = repeat_stats[repeat_stats['repeat_guest_label']=='Repeat Guest']
new_vals    = [float(new_row[m].iloc[0])    if len(new_row)    else 0 for m in metrics]
repeat_vals = [float(repeat_row[m].iloc[0]) if len(repeat_row) else 0 for m in metrics]

ax1.bar(x3 - w3/2, new_vals,    w3, label='New Guest',    color='steelblue', edgecolor='white')
ax1.bar(x3 + w3/2, repeat_vals, w3, label='Repeat Guest', color='goldenrod', edgecolor='white')
ax1.set_xticks(x3); ax1.set_xticklabels(metric_labels, fontsize=9)
ax1.set_title('Key Metrics: Repeat vs New Guests', fontweight='bold')
ax1.set_ylabel('Value'); ax1.legend(); ax1.grid(axis='y', alpha=0.4)

cancel_vals   = list(repeat_stats['cancel_rate'])
cancel_labels = list(repeat_stats['repeat_guest_label'])
bar_colors6   = ['steelblue' if 'New' in l else 'goldenrod' for l in cancel_labels]
bars6 = ax2.bar(cancel_labels, cancel_vals,
                color=bar_colors6, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars6, cancel_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')
ax2.set_title('Cancellation Rate: Repeat vs New Guests', fontweight='bold')
ax2.set_ylabel('Cancellation Rate (%)')
ax2.set_ylim(0, max(cancel_vals)*1.2); ax2.grid(axis='y', alpha=0.4)

plt.tight_layout()
p6 = os.path.join(EXPORT_DIR, 'chart6_repeat_vs_new.png')
plt.savefig(p6, dpi=150, bbox_inches='tight'); plt.close()
print(f"✅ Saved → {p6}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT CLEAN CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPORTING CLEAN DATASET")
print("="*60)

csv_path = os.path.join(EXPORT_DIR, 'hotel_bookings_clean.csv')
df.to_csv(csv_path, index=False)
print(f"  File      : {csv_path}")
print(f"  Rows      : {len(df):,}")
print(f"  Columns   : {len(df.columns)}")
print("✅ Clean dataset ready for Power BI dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  HOTEL BOOKING INTELLIGENCE — EXECUTIVE SUMMARY")
print("  Business Analyst Portfolio | Sabre-Style Analysis")
print("="*65)

df_conf_f = df[df['is_canceled']==0].copy()
df_canc_f = df[df['is_canceled']==1].copy()

total_bk    = len(df)
city_pct    = (df['hotel']=='City Hotel').sum() / total_bk * 100
resort_pct  = (df['hotel']=='Resort Hotel').sum() / total_bk * 100
overall_cr  = df['is_canceled'].mean() * 100
city_cr     = df[df['hotel']=='City Hotel']['is_canceled'].mean() * 100
resort_cr   = df[df['hotel']=='Resort Hotel']['is_canceled'].mean() * 100
peak_mon    = (df.groupby('arrival_date_month')['is_canceled'].mean()*100).idxmax()
rev_lost    = df_canc_f['revenue'].sum()
total_rev   = df_conf_f['revenue'].sum()
avg_adr_f   = df_conf_f['adr'].mean()
top_seg_f   = df_conf_f.groupby('market_segment')['revenue'].sum().idxmax()
top_cty_f   = df_conf_f.groupby('country')['revenue'].sum().idxmax()
avg_rev_f   = df_conf_f['revenue'].mean()
family_pct  = df['is_family'].mean()*100
repeat_pct  = df['is_repeated_guest'].mean()*100
avg_lead_f  = df['lead_time'].mean()
avg_nights_f= df['total_nights'].mean()

print(f"""
DATASET OVERVIEW
----------------------------------------------------------------
  Total bookings analysed  : {total_bk:,}
  City Hotel               : {city_pct:.1f}% of bookings
  Resort Hotel             : {resort_pct:.1f}% of bookings

CANCELLATION ANALYSIS
----------------------------------------------------------------
  Overall cancellation rate: {overall_cr:.1f}%
  City Hotel cancel rate   : {city_cr:.1f}%
  Resort Hotel cancel rate : {resort_cr:.1f}%
  Peak cancellation month  : {peak_mon}
  Revenue lost (cancelled) : €{rev_lost:,.0f}

REVENUE ANALYSIS
----------------------------------------------------------------
  Total confirmed revenue  : €{total_rev:,.0f}
  Average ADR (confirmed)  : €{avg_adr_f:.2f} / night
  Top market segment       : {top_seg_f}
  Top country by revenue   : {top_cty_f}
  Avg revenue per booking  : €{avg_rev_f:.2f}

GUEST INSIGHTS
----------------------------------------------------------------
  Family bookings          : {family_pct:.1f}%
  Repeat guests            : {repeat_pct:.1f}%
  Average lead time        : {avg_lead_f:.0f} days
  Average stay length      : {avg_nights_f:.1f} nights
""")

print("="*65)
print()
print("🎉 PROJECT COMPLETE — All 6 charts exported, clean CSV ready")
print("   for Power BI, README created.")
print("   Your Sabre BA portfolio project is ready.")
print("="*65)

# List all exported files
print("\n📁 Files in exports/:")
for f_name in sorted(os.listdir(EXPORT_DIR)):
    size = os.path.getsize(os.path.join(EXPORT_DIR, f_name))
    print(f"   {f_name:<45} {size/1024:.1f} KB")
