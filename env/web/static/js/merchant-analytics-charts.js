(function(global) {
  "use strict";

  const metricLabels = {
    orders: "Orders",
    gmv: "GMV",
    gross_profit: "Gross profit",
    net_profit: "Net profit",
    fine: "Fines",
    late_count: "Late orders",
    stockout_count: "Stockout orders",
    refund_count: "Refund orders",
    bad_review_count: "Bad-review orders",
  };

  function cloneSeries(value) {
    return Array.isArray(value) ? value.map(point => Array.isArray(point) ? point.slice() : point) : [];
  }

  function values(value) {
    return Array.isArray(value) ? value.map(item => Number(item || 0)) : [];
  }

  function normalizeDashboardProducts(productPayload) {
    return (productPayload?.series || []).map(product => {
      const data = Array.isArray(product.data) ? product.data : [];
      const field = name => data.map(point => Number(point?.[name] || 0));
      return {
        product_id: product.product_id,
        name: product.name || product.product_id,
        category: product.category || "",
        orders: field("orders"),
        gmv: field("gmv"),
        gross_profit: field("gross_profit"),
        net_profit: field("net_profit"),
        fine: field("fine"),
        late_count: field("late_count"),
        stockout_count: field("stockout_count"),
        refund_count: field("refund_count"),
        bad_review_count: field("bad_review_count"),
        order_anomalies: field("order_anomalies"),
        supply_chain_anomalies: field("supply_chain_anomalies"),
      };
    });
  }

  function fromDashboardPayload(payload) {
    const series = payload?.series || {};
    const listingOps = payload?.listing_ops?.series || {};
    const productPayload = payload?.daily_sales_by_product || {};
    const bucketMeta = (productPayload.buckets || []).length
      ? productPayload.buckets.map(bucket => ({...bucket}))
      : (productPayload.days || []).map(day => ({
        key: `D${day}`,
        label: `D${day}`,
        start_day: Number(day),
        end_day: Number(day),
        day: Number(day),
      }));
    const buckets = bucketMeta.map(bucket => String(bucket.label || bucket.key));
    return {
      buckets,
      bucketMeta,
      assets: {
        balance: cloneSeries(series.balance),
        deposit_pool: cloneSeries(series.deposit_pool),
        in_transit: cloneSeries(series.in_transit),
        receivable: cloneSeries(series.receivable),
        net_assets: cloneSeries(series.net_assets),
      },
      listings: {
        active: cloneSeries(series.n_active_listings),
        ops: {
          total: cloneSeries(listingOps.ops),
          list: cloneSeries(listingOps.list),
          delist: cloneSeries(listingOps.delist),
          price: cloneSeries(listingOps.price),
        },
      },
      pnl: {
        gmv: cloneSeries(series.cum_gmv),
        cost: cloneSeries(series.cum_cost),
        gross_profit: cloneSeries(series.cum_gross_profit),
        net_profit: cloneSeries(series.cum_net_profit),
        fine: cloneSeries(series.cum_fine),
      },
      rating: cloneSeries(series.shop_rating_mean?.length ? series.shop_rating_mean : series.shop_rating_score),
      products: normalizeDashboardProducts(productPayload),
    };
  }

  function productTotals(products, metric) {
    return (products || []).map(product => ({
      product_id: product.product_id,
      name: product.name || product.product_id,
      category: product.category || "",
      value: values(product[metric]).reduce((sum, value) => sum + value, 0),
    })).sort((a, b) => b.value - a.value || String(a.product_id).localeCompare(String(b.product_id)));
  }

  function pointValues(data, keepPoints) {
    const cloned = cloneSeries(data);
    return keepPoints
      ? cloned
      : cloned.map(point => Number(Array.isArray(point) ? point[1] || 0 : point || 0));
  }

  function pointLabels(data) {
    return (data || []).map((point, index) => (
      Array.isArray(point) ? String(point[0]) : String(index + 1)
    ));
  }

  function defaultLineSeries(name, data, color, options = {}) {
    return {
      name,
      type: "line",
      data: pointValues(data, Boolean(options.keepPoints)),
      symbol: "none",
      smooth: options.smooth ?? .18,
      lineStyle: {width: options.width || 2, color},
      itemStyle: {color},
      areaStyle: options.area ? {color, opacity: .08} : undefined,
      yAxisIndex: options.yAxisIndex || 0,
      ...(options.extra || {}),
    };
  }

  function defaultBarSeries(name, data, color, options = {}) {
    return {
      name,
      type: "bar",
      data: pointValues(data, Boolean(options.keepPoints)),
      yAxisIndex: options.yAxisIndex || 0,
      stack: options.stack,
      barMaxWidth: options.barMaxWidth || 12,
      itemStyle: {color},
      ...(options.extra || {}),
    };
  }

  function seriesFactory(options, type) {
    return type === "bar"
      ? (options.barFactory || defaultBarSeries)
      : (options.lineFactory || defaultLineSeries);
  }

  function buildAssetsOption(model, options = {}) {
    const labels = {
      balance: "balance",
      deposit_pool: "deposit_pool",
      in_transit: "in_transit",
      receivable: "receivable",
      net_assets: "net_assets",
      ...(options.labels || {}),
    };
    const colors = {
      balance: "#147a4b",
      deposit_pool: "#769b84",
      in_transit: "#d49138",
      receivable: "#296b91",
      net_assets: "#17211b",
      ...(options.colors || {}),
    };
    const line = seriesFactory(options, "line");
    const lineOptions = key => ({
      keepPoints: options.keepPoints,
      ...(options.lineOptions?.[key] || {}),
    });
    const data = model?.assets || {};
    const order = options.order || [
      "balance", "deposit_pool", "in_transit", "receivable", "net_assets",
    ];
    return {
      xAxis: {data: pointLabels(model?.assets?.net_assets)},
      series: order.map(key => line(
        labels[key], data[key], colors[key], lineOptions(key),
      )),
    };
  }

  function buildListingsOption(model, options = {}) {
    const labels = {
      active: "n_active",
      total: "ops",
      list: "list",
      delist: "delist",
      price: "price",
      ...(options.labels || {}),
    };
    const colors = {
      active: "#147a4b",
      total: "#147a4b",
      list: "#75ad8c",
      delist: "#d27672",
      price: "#d7a85f",
      ...(options.colors || {}),
    };
    const line = seriesFactory(options, "line");
    const bar = seriesFactory(options, "bar");
    const keepPoints = options.keepPoints;
    const mode = options.mode || "combined";
    const lineFor = (key, data, extra = {}) => line(
      labels[key], data, colors[key], {keepPoints, ...extra},
    );
    const barFor = (key, data) => bar(labels[key], data, colors[key], {
      keepPoints,
      yAxisIndex: 1,
      stack: "ops",
      barMaxWidth: 10,
    });
    let series;
    if (mode === "ops") {
      series = [lineFor("total", model?.listings?.ops?.total, {area: true})];
    } else if (mode === "ops_breakdown") {
      series = [
        lineFor("list", model?.listings?.ops?.list),
        lineFor("delist", model?.listings?.ops?.delist),
        lineFor("price", model?.listings?.ops?.price),
      ];
    } else if (mode === "active") {
      series = [lineFor("active", model?.listings?.active, {area: true})];
    } else {
      series = [
        lineFor("active", model?.listings?.active, {area: true}),
        barFor("list", model?.listings?.ops?.list),
        barFor("delist", model?.listings?.ops?.delist),
        barFor("price", model?.listings?.ops?.price),
      ];
    }
    return {
      xAxis: {data: pointLabels(model?.listings?.active)},
      series,
    };
  }

  function buildPnlOption(model, options = {}) {
    const labels = {
      gmv: "cum_gmv",
      cost: "cum_cost",
      gross_profit: "cum_gross_profit",
      net_profit: "cum_net_profit",
      fine: "cum_fine",
      ...(options.labels || {}),
    };
    const colors = {
      gmv: "#296b91",
      cost: "#8d9891",
      gross_profit: "#74a488",
      net_profit: "#147a4b",
      fine: "#b13a39",
      ...(options.colors || {}),
    };
    const line = seriesFactory(options, "line");
    const lineOptions = key => ({
      keepPoints: options.keepPoints,
      ...(options.lineOptions?.[key] || {}),
    });
    return {
      xAxis: {data: pointLabels(model?.pnl?.net_profit)},
      series: [
        line(labels.gmv, model?.pnl?.gmv, colors.gmv, lineOptions("gmv")),
        line(labels.cost, model?.pnl?.cost, colors.cost, lineOptions("cost")),
        line(labels.gross_profit, model?.pnl?.gross_profit, colors.gross_profit, lineOptions("gross_profit")),
        line(labels.net_profit, model?.pnl?.net_profit, colors.net_profit, lineOptions("net_profit")),
        line(labels.fine, model?.pnl?.fine, colors.fine, lineOptions("fine")),
      ],
    };
  }

  function ratingAxisBounds(data) {
    const observed = (data || []).map(point => Number(
      Array.isArray(point) ? point[1] : point
    )).filter(Number.isFinite);
    if (!observed.length) return {min: 1, max: 5};
    const low = Math.min(...observed);
    const high = Math.max(...observed);
    const dataSpan = Math.max(0, high - low);
    const padding = Math.max(0.05, dataSpan * 0.18);
    const targetSpan = Math.min(4, Math.max(0.2, dataSpan + padding * 2));
    const mid = (low + high) / 2;
    let min = mid - targetSpan / 2;
    let max = mid + targetSpan / 2;
    if (min < 1) {
      max = Math.min(5, max + (1 - min));
      min = 1;
    }
    if (max > 5) {
      min = Math.max(1, min - (max - 5));
      max = 5;
    }
    return {min, max};
  }

  function buildRatingOption(model, options = {}) {
    const line = seriesFactory(options, "line");
    return {
      xAxis: {data: pointLabels(model?.rating)},
      yAxis: ratingAxisBounds(model?.rating),
      series: [line(
        options.label || "shop_rating",
        model?.rating,
        options.color || "#d08b27",
        {
          keepPoints: options.keepPoints,
          width: 3,
          area: true,
          ...(options.lineOptions || {}),
        },
      )],
    };
  }

  function buildProductTrendOption(model, options = {}) {
    const selectedProductId = options.selectedProductId
      ? String(options.selectedProductId)
      : null;
    const products = options.products || model?.products || [];
    const selected = selectedProductId
      ? products.filter(product => String(product.product_id) === selectedProductId)
      : products;
    const metrics = options.metrics || [
      "orders", "gmv", "gross_profit", "net_profit", "fine",
      "late_count", "stockout_count", "refund_count", "bad_review_count",
    ];
    const colors = options.colors || [
      "#296b91", "#5c87a0", "#75a88a", "#147a4b", "#b13a39",
      "#d49138", "#bb665f", "#8a6eae", "#6c7788",
    ];
    const line = seriesFactory(options, "line");
    const bar = seriesFactory(options, "bar");
    const series = [];
    selected.forEach((product, productIndex) => {
      if (selectedProductId) {
        metrics.forEach((metric, index) => {
          const factory = index < 5 ? line : bar;
          series.push(factory(
            options.metricLabels?.[metric] || metricLabels[metric] || metric,
            product[metric] || [],
            colors[index % colors.length],
            {
              yAxisIndex: index < 5 ? 0 : 1,
              width: metric === "net_profit" ? 3 : 1.8,
              barMaxWidth: 10,
            },
          ));
        });
      } else {
        series.push(line(
          product.name || product.product_id,
          product.net_profit || [],
          colors[productIndex % colors.length],
          {width: 2},
        ));
      }
    });
    return {
      xAxis: {data: (model?.buckets || []).map(String)},
      selected,
      series,
    };
  }

  function buildProductDailyRankedOption(model, metric, options = {}) {
    const products = options.products || model?.products || [];
    const buckets = (model?.buckets || []).map(String);
    const palette = options.colors || [
      "#168AAD", "#2F9E44", "#F08C00", "#8A6EAE", "#B13A39",
      "#296B91", "#75A88A", "#D49138", "#6C7788", "#0F766E",
    ];
    const colorByProduct = new Map(products.map((product, index) => [
      String(product.product_id),
      typeof options.colorForProduct === "function"
        ? options.colorForProduct(product, index)
        : palette[index % palette.length],
    ]));
    const selectedProductId = options.selectedProductId
      ? String(options.selectedProductId)
      : null;
    const visibleValue = value => (
      metric === "gross_profit" || metric === "net_profit"
        ? value !== 0
        : value > 0
    );
    const bucketMeta = Array.isArray(model?.bucketMeta) ? model.bucketMeta : [];
    const byBucket = buckets.map((_bucket, bucketIndex) => products
      .map(product => {
        const productId = String(product.product_id);
        const value = Number(product?.[metric]?.[bucketIndex] || 0);
        const meta = bucketMeta[bucketIndex] || {};
        return {
          value,
          metric,
          bucket: buckets[bucketIndex],
          label: String(meta.label || meta.key || buckets[bucketIndex]),
          start_day: Number(meta.start_day ?? meta.day ?? 0),
          end_day: Number(meta.end_day ?? meta.day ?? 0),
          day: Number(meta.day ?? meta.start_day ?? 0),
          product_id: productId,
          product_name: product.name || productId,
          category: product.category || "",
          orders: Number(product.orders?.[bucketIndex] || 0),
          gmv: Number(product.gmv?.[bucketIndex] || 0),
          gross_profit: Number(product.gross_profit?.[bucketIndex] || 0),
          net_profit: Number(product.net_profit?.[bucketIndex] || 0),
          supply_chain_anomalies: Number(product.supply_chain_anomalies?.[bucketIndex] || 0),
          order_anomalies: Number(product.order_anomalies?.[bucketIndex] || 0),
          color: colorByProduct.get(productId),
        };
      })
      .filter(segment => visibleValue(segment.value))
      .sort((a, b) => b.value - a.value || a.product_id.localeCompare(b.product_id)));
    const depth = Math.max(0, ...byBucket.map(segments => segments.length));
    return {
      xAxis: {data: buckets},
      series: Array.from({length: depth}, (_unused, rank) => ({
        name: `rank-layer-${rank}`,
        type: "bar",
        stack: options.stack || "daily-products",
        barWidth: options.barWidth || 18,
        emphasis: {disabled: true},
        data: byBucket.map(segments => {
          const segment = segments[rank];
          if (!segment) return {value: 0, itemStyle: {opacity: 0}};
          const customStyle = typeof options.itemStyleForSegment === "function"
            ? options.itemStyleForSegment(segment) || {}
            : {};
          return {
            ...segment,
            itemStyle: {
              color: segment.value < 0 ? (options.negativeColor || "#E03131") : segment.color,
              opacity: selectedProductId && segment.product_id !== selectedProductId ? .28 : 1,
              borderColor: "rgba(15,23,32,.55)",
              borderWidth: 1,
              ...customStyle,
            },
          };
        }),
      })),
    };
  }

  function buildProductRankOption(model, metric, options = {}) {
    let totals = productTotals(model?.products || [], metric)
      .filter(item => (
        item.product_id !== null
        && item.product_id !== undefined
        && String(item.product_id).trim() !== ""
        && Number.isFinite(item.value)
        && item.value !== 0
      ))
      .slice(0, options.limit || 20);
    if (options.reverse !== false) totals = totals.reverse();
    const color = options.color || "#168AAD";
    return {
      totals,
      xAxis: {type: "value"},
      yAxis: {
        type: "category",
        data: totals.map(item => item.name),
      },
      series: [{
        name: options.label || metricLabels[metric] || metric,
        type: "bar",
        data: totals.map(item => ({
          value: item.value,
          product_id: item.product_id,
          itemStyle: {color},
        })),
        barMaxWidth: options.barMaxWidth || 18,
      }],
    };
  }

  const optionBuilders = Object.freeze({
    assets: buildAssetsOption,
    listings: buildListingsOption,
    pnl: buildPnlOption,
    rating: buildRatingOption,
    productTrend: buildProductTrendOption,
    productDailyRanked: buildProductDailyRankedOption,
    productRank: buildProductRankOption,
  });

  global.MerchantBenchMerchantCharts = Object.freeze({
    metricLabels,
    fromDashboardPayload,
    productTotals,
    ratingAxisBounds,
    normalizeDashboardProducts,
    optionBuilders,
  });
})(window);
