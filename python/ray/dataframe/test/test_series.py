from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import pandas as pd

import ray.dataframe as rdf
from ray.dataframe.utils import to_pandas_series


@pytest.fixture
def ray_series_equals_pandas(ray_series, pandas_series):
    return to_pandas_series(ray_series).sort_index() \
        .equals(pandas_series.sort_index())


def test_int_series():
    data = [1, 4, -2, 3, -12, 23, 1, 3, 12]

    pandas_series = pd.Series(data)
    # TODO: Implement index onto series
    ray_series = rdf.Series(data, index=pandas_series.index)

    test_T(ray_series, pandas_series)
    test___abs__(ray_series, pandas_series)
    test___add__(ray_series, pandas_series)
    test___and__(ray_series, pandas_series)
    test___array__(ray_series, pandas_series)
    test___array_prepare__(ray_series, pandas_series)
    test___array_priority__(ray_series, pandas_series)
    test___array_wrap__(ray_series, pandas_series)
    test___bool__(ray_series, pandas_series)
    test___bytes__(ray_series, pandas_series)
    test___class__(ray_series, pandas_series)
    test___contains__(ray_series, pandas_series)
    test___copy__(ray_series, pandas_series)
    test___deepcopy__(ray_series, pandas_series)
    test___delitem__(ray_series, pandas_series)
    test___div__(ray_series, pandas_series)
    test___divmod__(ray_series, pandas_series)
    test___doc__(ray_series, pandas_series)
    test___eq__(ray_series, pandas_series)
    test___finalize__(ray_series, pandas_series)
    test___float__(ray_series, pandas_series)
    test___floordiv__(ray_series, pandas_series)
    test___ge__(ray_series, pandas_series)
    test___getitem__(ray_series, pandas_series)
    test___getstate__(ray_series, pandas_series)
    test___gt__(ray_series, pandas_series)
    test___iadd__(ray_series, pandas_series)
    test___imul__(ray_series, pandas_series)
    test___int__(ray_series, pandas_series)
    test___invert__(ray_series, pandas_series)
    test___ipow__(ray_series, pandas_series)
    test___isub__(ray_series, pandas_series)
    test___iter__(ray_series, pandas_series)
    test___itruediv__(ray_series, pandas_series)
    test___le__(ray_series, pandas_series)
    test___len__(ray_series, pandas_series)
    test___long__(ray_series, pandas_series)
    test___lt__(ray_series, pandas_series)
    test___mod__(ray_series, pandas_series)
    test___mul__(ray_series, pandas_series)
    test___ne__(ray_series, pandas_series)
    test___neg__(ray_series, pandas_series)
    test___nonzero__(ray_series, pandas_series)
    test___or__(ray_series, pandas_series)
    test___pow__(ray_series, pandas_series)
    test___repr__(ray_series, pandas_series)
    test___round__(ray_series, pandas_series)
    test___setitem__(ray_series, pandas_series)
    test___setstate__(ray_series, pandas_series)
    test___sizeof__(ray_series, pandas_series)
    test___str__(ray_series, pandas_series)
    test___sub__(ray_series, pandas_series)
    test___truediv__(ray_series, pandas_series)
    test___xor__(ray_series, pandas_series)
    test_abs(ray_series, pandas_series)
    test_add(ray_series, pandas_series)
    test_add_prefix(ray_series, pandas_series)
    test_add_suffix(ray_series, pandas_series)
    test_agg(ray_series, pandas_series)
    test_aggregate(ray_series, pandas_series)
    test_align(ray_series, pandas_series)
    test_all(ray_series, pandas_series)
    test_any(ray_series, pandas_series)
    test_append(ray_series, pandas_series)
    test_apply(ray_series, pandas_series)
    test_argmax(ray_series, pandas_series)
    test_argmin(ray_series, pandas_series)
    test_argsort(ray_series, pandas_series)
    test_as_blocks(ray_series, pandas_series)
    test_as_matrix(ray_series, pandas_series)
    test_asfreq(ray_series, pandas_series)
    test_asobject(ray_series, pandas_series)
    test_asof(ray_series, pandas_series)
    test_astype(ray_series, pandas_series)
    test_at(ray_series, pandas_series)
    test_at_time(ray_series, pandas_series)
    test_autocorr(ray_series, pandas_series)
    test_axes(ray_series, pandas_series)
    test_base(ray_series, pandas_series)
    test_between(ray_series, pandas_series)
    test_between_time(ray_series, pandas_series)
    test_bfill(ray_series, pandas_series)
    test_blocks(ray_series, pandas_series)
    test_bool(ray_series, pandas_series)
    test_clip(ray_series, pandas_series)
    test_clip_lower(ray_series, pandas_series)
    test_clip_upper(ray_series, pandas_series)
    test_combine(ray_series, pandas_series)
    test_combine_first(ray_series, pandas_series)
    test_compound(ray_series, pandas_series)
    test_compress(ray_series, pandas_series)
    test_consolidate(ray_series, pandas_series)
    test_convert_objects(ray_series, pandas_series)
    test_copy(ray_series, pandas_series)
    test_corr(ray_series, pandas_series)
    test_count(ray_series, pandas_series)
    test_cov(ray_series, pandas_series)
    test_cummax(ray_series, pandas_series)
    test_cummin(ray_series, pandas_series)
    test_cumprod(ray_series, pandas_series)
    test_cumsum(ray_series, pandas_series)
    test_data(ray_series, pandas_series)
    test_describe(ray_series, pandas_series)
    test_diff(ray_series, pandas_series)
    test_div(ray_series, pandas_series)
    test_divide(ray_series, pandas_series)
    test_dot(ray_series, pandas_series)
    test_drop(ray_series, pandas_series)
    test_drop_duplicates(ray_series, pandas_series)
    test_dropna(ray_series, pandas_series)
    test_dtype(ray_series, pandas_series)
    test_dtypes(ray_series, pandas_series)
    test_duplicated(ray_series, pandas_series)
    test_empty(ray_series, pandas_series)
    test_eq(ray_series, pandas_series)
    test_equals(ray_series, pandas_series)
    test_ewm(ray_series, pandas_series)
    test_expanding(ray_series, pandas_series)
    test_factorize(ray_series, pandas_series)
    test_ffill(ray_series, pandas_series)
    test_fillna(ray_series, pandas_series)
    test_filter(ray_series, pandas_series)
    test_first(ray_series, pandas_series)
    test_first_valid_index(ray_series, pandas_series)
    test_flags(ray_series, pandas_series)
    test_floordiv(ray_series, pandas_series)
    test_from_array(ray_series, pandas_series)
    test_from_csv(ray_series, pandas_series)
    test_ftype(ray_series, pandas_series)
    test_ftypes(ray_series, pandas_series)
    test_ge(ray_series, pandas_series)
    test_get(ray_series, pandas_series)
    test_get_dtype_counts(ray_series, pandas_series)
    test_get_ftype_counts(ray_series, pandas_series)
    test_get_value(ray_series, pandas_series)
    test_get_values(ray_series, pandas_series)
    test_groupby(ray_series, pandas_series)
    test_gt(ray_series, pandas_series)
    test_hasnans(ray_series, pandas_series)
    test_head(ray_series, pandas_series)
    test_hist(ray_series, pandas_series)
    test_iat(ray_series, pandas_series)
    test_idxmax(ray_series, pandas_series)
    test_idxmin(ray_series, pandas_series)
    test_iloc(ray_series, pandas_series)
    test_imag(ray_series, pandas_series)
    test_index(ray_series, pandas_series)
    test_interpolate(ray_series, pandas_series)
    test_is_copy(ray_series, pandas_series)
    test_is_monotonic(ray_series, pandas_series)
    test_is_monotonic_decreasing(ray_series, pandas_series)
    test_is_monotonic_increasing(ray_series, pandas_series)
    test_is_unique(ray_series, pandas_series)
    test_isin(ray_series, pandas_series)
    test_isnull(ray_series, pandas_series)
    test_item(ray_series, pandas_series)
    test_items(ray_series, pandas_series)
    test_itemsize(ray_series, pandas_series)
    test_iteritems(ray_series, pandas_series)
    test_ix(ray_series, pandas_series)
    test_keys(ray_series, pandas_series)
    test_kurt(ray_series, pandas_series)
    test_kurtosis(ray_series, pandas_series)
    test_last(ray_series, pandas_series)
    test_last_valid_index(ray_series, pandas_series)
    test_le(ray_series, pandas_series)
    test_loc(ray_series, pandas_series)
    test_lt(ray_series, pandas_series)
    test_mad(ray_series, pandas_series)
    test_map(ray_series, pandas_series)
    test_mask(ray_series, pandas_series)
    test_max(ray_series, pandas_series)
    test_mean(ray_series, pandas_series)
    test_median(ray_series, pandas_series)
    test_memory_usage(ray_series, pandas_series)
    test_min(ray_series, pandas_series)
    test_mod(ray_series, pandas_series)
    test_mode(ray_series, pandas_series)
    test_mul(ray_series, pandas_series)
    test_multiply(ray_series, pandas_series)
    test_name(ray_series, pandas_series)
    test_nbytes(ray_series, pandas_series)
    test_ndim(ray_series, pandas_series)
    test_ne(ray_series, pandas_series)
    test_nlargest(ray_series, pandas_series)
    test_nonzero(ray_series, pandas_series)
    test_notnull(ray_series, pandas_series)
    test_nsmallest(ray_series, pandas_series)
    test_nunique(ray_series, pandas_series)
    test_pct_change(ray_series, pandas_series)
    test_pipe(ray_series, pandas_series)
    test_plot(ray_series, pandas_series)
    test_pop(ray_series, pandas_series)
    test_pow(ray_series, pandas_series)
    test_prod(ray_series, pandas_series)
    test_product(ray_series, pandas_series)
    test_ptp(ray_series, pandas_series)
    test_put(ray_series, pandas_series)
    test_quantile(ray_series, pandas_series)
    test_radd(ray_series, pandas_series)
    test_rank(ray_series, pandas_series)
    test_ravel(ray_series, pandas_series)
    test_rdiv(ray_series, pandas_series)
    test_real(ray_series, pandas_series)
    test_reindex(ray_series, pandas_series)
    test_reindex_axis(ray_series, pandas_series)
    test_reindex_like(ray_series, pandas_series)
    test_rename(ray_series, pandas_series)
    test_rename_axis(ray_series, pandas_series)
    test_reorder_levels(ray_series, pandas_series)
    test_repeat(ray_series, pandas_series)
    test_replace(ray_series, pandas_series)
    test_resample(ray_series, pandas_series)
    test_reset_index(ray_series, pandas_series)
    test_reshape(ray_series, pandas_series)
    test_rfloordiv(ray_series, pandas_series)
    test_rmod(ray_series, pandas_series)
    test_rmul(ray_series, pandas_series)
    test_rolling(ray_series, pandas_series)
    test_round(ray_series, pandas_series)
    test_rpow(ray_series, pandas_series)
    test_rsub(ray_series, pandas_series)
    test_rtruediv(ray_series, pandas_series)
    test_sample(ray_series, pandas_series)
    test_searchsorted(ray_series, pandas_series)
    test_select(ray_series, pandas_series)
    test_sem(ray_series, pandas_series)
    test_set_axis(ray_series, pandas_series)
    test_set_value(ray_series, pandas_series)
    test_shape(ray_series, pandas_series)
    test_shift(ray_series, pandas_series)
    test_size(ray_series, pandas_series)
    test_skew(ray_series, pandas_series)
    test_slice_shift(ray_series, pandas_series)
    test_sort_index(ray_series, pandas_series)
    test_sort_values(ray_series, pandas_series)
    test_sortlevel(ray_series, pandas_series)
    test_squeeze(ray_series, pandas_series)
    test_std(ray_series, pandas_series)
    test_strides(ray_series, pandas_series)
    test_sub(ray_series, pandas_series)
    test_subtract(ray_series, pandas_series)
    test_sum(ray_series, pandas_series)
    test_swapaxes(ray_series, pandas_series)
    test_swaplevel(ray_series, pandas_series)
    test_tail(ray_series, pandas_series)
    test_take(ray_series, pandas_series)
    test_to_clipboard(ray_series, pandas_series)
    test_to_csv(ray_series, pandas_series)
    test_to_dense(ray_series, pandas_series)
    test_to_dict(ray_series, pandas_series)
    test_to_excel(ray_series, pandas_series)
    test_to_frame(ray_series, pandas_series)
    test_to_hdf(ray_series, pandas_series)
    test_to_json(ray_series, pandas_series)
    test_to_latex(ray_series, pandas_series)
    test_to_msgpack(ray_series, pandas_series)
    test_to_period(ray_series, pandas_series)
    test_to_pickle(ray_series, pandas_series)
    test_to_sparse(ray_series, pandas_series)
    test_to_sql(ray_series, pandas_series)
    test_to_string(ray_series, pandas_series)
    test_to_timestamp(ray_series, pandas_series)
    test_to_xarray(ray_series, pandas_series)
    test_tolist(ray_series, pandas_series)
    test_transform(ray_series, pandas_series)
    test_transpose(ray_series, pandas_series)
    test_truediv(ray_series, pandas_series)
    test_truncate(ray_series, pandas_series)
    test_tshift(ray_series, pandas_series)
    test_tz_convert(ray_series, pandas_series)
    test_tz_localize(ray_series, pandas_series)
    test_unique(ray_series, pandas_series)
    test_unstack(ray_series, pandas_series)
    test_update(ray_series, pandas_series)
    test_valid(ray_series, pandas_series)
    test_value_counts(ray_series, pandas_series)
    test_values(ray_series, pandas_series)
    test_var(ray_series, pandas_series)
    test_view(ray_series, pandas_series)
    test_where(ray_series, pandas_series)
    test_xs(ray_series, pandas_series)


@pytest.fixture
def test_T(ray_series, pandas_series):
    assert ray_series_equals_pandas(ray_series.T, pandas_series.T)


@pytest.fixture
def test___abs__(ray_series, pandas_series):
    assert ray_series_equals_pandas(abs(ray_series), abs(pandas_series.T))


@pytest.fixture
def test___add__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__add__(None, None)


@pytest.fixture
def test___and__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__and__(None)


@pytest.fixture
def test___array__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__array__(None)


@pytest.fixture
def test___array_prepare__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__array_prepare__(None)


@pytest.fixture
def test___array_priority__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__array_priority__


@pytest.fixture
def test___array_wrap__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__array_wrap__(None)


@pytest.fixture
def test___bool__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__bool__()


@pytest.fixture
def test___bytes__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__bytes__()


@pytest.fixture
def test___class__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__class__(None, None, None, None, None)


@pytest.fixture
def test___contains__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__contains__(None)


@pytest.fixture
def test___copy__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__copy__(None)


@pytest.fixture
def test___deepcopy__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__deepcopy__(None)


@pytest.fixture
def test___delitem__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__delitem__(None)


@pytest.fixture
def test___div__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__div__(None, None)


@pytest.fixture
def test___divmod__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__divmod__(None, None)


@pytest.fixture
def test___doc__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__doc__


@pytest.fixture
def test___eq__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__eq__(None)


@pytest.fixture
def test___finalize__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__finalize__(None, None)


@pytest.fixture
def test___float__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__float__()


@pytest.fixture
def test___floordiv__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__floordiv__(None, None)


@pytest.fixture
def test___ge__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__ge__(None)


@pytest.fixture
def test___getitem__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__getitem__(None)


@pytest.fixture
def test___getstate__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__getstate__()


@pytest.fixture
def test___gt__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__gt__(None)


@pytest.fixture
def test___iadd__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__iadd__(None)


@pytest.fixture
def test___imul__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__imul__(None)


@pytest.fixture
def test___int__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__int__()


@pytest.fixture
def test___invert__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__invert__()


@pytest.fixture
def test___ipow__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__ipow__(None)


@pytest.fixture
def test___isub__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__isub__(None)


@pytest.fixture
def test___iter__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__iter__()


@pytest.fixture
def test___itruediv__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__itruediv__(None)


@pytest.fixture
def test___le__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__le__(None)


@pytest.fixture
def test___len__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__len__()


@pytest.fixture
def test___long__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__long__()


@pytest.fixture
def test___lt__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__lt__(None)


@pytest.fixture
def test___mod__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__mod__(None, None)


@pytest.fixture
def test___mul__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__mul__(None, None)


@pytest.fixture
def test___ne__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__ne__(None)


@pytest.fixture
def test___neg__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__neg__()


@pytest.fixture
def test___nonzero__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__nonzero__()


@pytest.fixture
def test___or__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__or__(None)


@pytest.fixture
def test___pow__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__pow__(None, None)


@pytest.fixture
def test___repr__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__repr__()


@pytest.fixture
def test___round__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__round__(None)


@pytest.fixture
def test___setitem__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__setitem__(None, None)


@pytest.fixture
def test___setstate__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__setstate__(None)


@pytest.fixture
def test___sizeof__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__sizeof__()


@pytest.fixture
def test___str__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__str__()


@pytest.fixture
def test___sub__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__sub__(None, None)


@pytest.fixture
def test___truediv__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__truediv__(None, None)


@pytest.fixture
def test___xor__(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.__xor__(None)


@pytest.fixture
def test_abs(ray_series, pandas_series):
    assert ray_series_equals_pandas(ray_series.abs(), pandas_series.abs())


@pytest.fixture
def test_add(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.add(None, None, None)


@pytest.fixture
def test_add_prefix(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.add_prefix(None)


@pytest.fixture
def test_add_suffix(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.add_suffix(None)


@pytest.fixture
def test_agg(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.agg(None, None, None)


@pytest.fixture
def test_aggregate(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.aggregate(None, None, None)


@pytest.fixture
def test_align(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.align(None, None, None, None, None, None, None, None, None)


@pytest.fixture
def test_all(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.all(None, None, None, None)


@pytest.fixture
def test_any(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.any(None, None, None, None)


@pytest.fixture
def test_append(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.append(None, None)


@pytest.fixture
def test_apply(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.apply(None, None, None)


@pytest.fixture
def test_argmax(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.argmax(None, None, None)


@pytest.fixture
def test_argmin(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.argmin(None, None, None)


@pytest.fixture
def test_argsort(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.argsort(None, None)


@pytest.fixture
def test_as_blocks(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.as_blocks(None)


@pytest.fixture
def test_as_matrix(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.as_matrix(None)


@pytest.fixture
def test_asfreq(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.asfreq(None, None, None, None)


@pytest.fixture
def test_asobject(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.asobject


@pytest.fixture
def test_asof(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.asof(None)


@pytest.fixture
def test_astype(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.astype(None, None, None)


@pytest.fixture
def test_at(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.at(None)


@pytest.fixture
def test_at_time(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.at_time(None)


@pytest.fixture
def test_autocorr(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.autocorr(None)


@pytest.fixture
def test_axes(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.axes


@pytest.fixture
def test_base(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.base


@pytest.fixture
def test_between(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.between(None, None)


@pytest.fixture
def test_between_time(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.between_time(None, None, None)


@pytest.fixture
def test_bfill(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.bfill(None, None, None)


@pytest.fixture
def test_blocks(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.blocks


@pytest.fixture
def test_bool(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.bool()


@pytest.fixture
def test_clip(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.clip(None, None, None, None)


@pytest.fixture
def test_clip_lower(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.clip_lower(None)


@pytest.fixture
def test_clip_upper(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.clip_upper(None)


@pytest.fixture
def test_combine(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.combine(None, None)


@pytest.fixture
def test_combine_first(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.combine_first(None)


@pytest.fixture
def test_compound(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.compound(None, None)


@pytest.fixture
def test_compress(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.compress(None, None)


@pytest.fixture
def test_consolidate(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.consolidate(None)


@pytest.fixture
def test_convert_objects(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.convert_objects(None, None, None)


@pytest.fixture
def test_copy(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.copy(None)


@pytest.fixture
def test_corr(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.corr(None, None)


@pytest.fixture
def test_count(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.count(None)


@pytest.fixture
def test_cov(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.cov(None)


@pytest.fixture
def test_cummax(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.cummax(None, None, None)


@pytest.fixture
def test_cummin(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.cummin(None, None, None)


@pytest.fixture
def test_cumprod(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.cumprod(None, None, None)


@pytest.fixture
def test_cumsum(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.cumsum(None, None, None)


@pytest.fixture
def test_data(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.data


@pytest.fixture
def test_describe(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.describe(None, None)


@pytest.fixture
def test_diff(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.diff(None)


@pytest.fixture
def test_div(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.div(None, None, None)


@pytest.fixture
def test_divide(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.divide(None, None, None)


@pytest.fixture
def test_dot(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.dot(None)


@pytest.fixture
def test_drop(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.drop(None, None, None, None)


@pytest.fixture
def test_drop_duplicates(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.drop_duplicates(None)


@pytest.fixture
def test_dropna(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.dropna(None, None)


@pytest.fixture
def test_dtype(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.dtype


@pytest.fixture
def test_dtypes(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.dtypes


@pytest.fixture
def test_duplicated(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.duplicated(None)


@pytest.fixture
def test_empty(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.empty


@pytest.fixture
def test_eq(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.eq(None, None, None)


@pytest.fixture
def test_equals(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.equals(None)


@pytest.fixture
def test_ewm(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ewm(None, None, None, None, None, None, None, None)


@pytest.fixture
def test_expanding(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.expanding(None, None, None)


@pytest.fixture
def test_factorize(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.factorize(None)


@pytest.fixture
def test_ffill(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ffill(None, None, None)


@pytest.fixture
def test_fillna(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.fillna(None, None, None, None, None, None)


@pytest.fixture
def test_filter(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.filter(None, None, None)


@pytest.fixture
def test_first(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.first(None)


@pytest.fixture
def test_first_valid_index(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.first_valid_index()


@pytest.fixture
def test_flags(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.flags


@pytest.fixture
def test_floordiv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.floordiv(None, None, None)


@pytest.fixture
def test_from_array(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.from_array(None, None, None, None, None)


@pytest.fixture
def test_from_csv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.from_csv(None, None, None, None, None, None)


@pytest.fixture
def test_ftype(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ftype


@pytest.fixture
def test_ftypes(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ftypes


@pytest.fixture
def test_ge(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ge(None, None, None)


@pytest.fixture
def test_get(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.get(None)


@pytest.fixture
def test_get_dtype_counts(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.get_dtype_counts()


@pytest.fixture
def test_get_ftype_counts(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.get_ftype_counts()


@pytest.fixture
def test_get_value(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.get_value(None)


@pytest.fixture
def test_get_values(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.get_values()


@pytest.fixture
def test_groupby(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.groupby(None, None, None, None, None, None, None)


@pytest.fixture
def test_gt(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.gt(None, None, None)


@pytest.fixture
def test_hasnans(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.hasnans


@pytest.fixture
def test_head(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.head(None)


@pytest.fixture
def test_hist(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.hist(None, None, None, None, None, None, None, None, None)


@pytest.fixture
def test_iat(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.iat(None)


@pytest.fixture
def test_idxmax(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.idxmax(None, None, None)


@pytest.fixture
def test_idxmin(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.idxmin(None, None, None)


@pytest.fixture
def test_iloc(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.iloc(None)


@pytest.fixture
def test_imag(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.imag


@pytest.fixture
def test_index(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.index


@pytest.fixture
def test_interpolate(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.interpolate(None, None, None, None, None, None)


@pytest.fixture
def test_is_copy(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.is_copy


@pytest.fixture
def test_is_monotonic(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic


@pytest.fixture
def test_is_monotonic_decreasing(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic_decreasing


@pytest.fixture
def test_is_monotonic_increasing(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic_increasing


@pytest.fixture
def test_is_unique(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.is_unique


@pytest.fixture
def test_isin(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.isin(None)


@pytest.fixture
def test_isnull(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.isnull()


@pytest.fixture
def test_item(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.item()


@pytest.fixture
def test_items(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.items()


@pytest.fixture
def test_itemsize(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.itemsize


@pytest.fixture
def test_iteritems(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.iteritems()


@pytest.fixture
def test_ix(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ix(None)


@pytest.fixture
def test_keys(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.keys()


@pytest.fixture
def test_kurt(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.kurt(None, None, None, None)


@pytest.fixture
def test_kurtosis(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.kurtosis(None, None, None, None)


@pytest.fixture
def test_last(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.last(None)


@pytest.fixture
def test_last_valid_index(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.last_valid_index()


@pytest.fixture
def test_le(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.le(None, None, None)


@pytest.fixture
def test_loc(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.loc(None)


@pytest.fixture
def test_lt(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.lt(None, None, None)


@pytest.fixture
def test_mad(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mad(None, None)


@pytest.fixture
def test_map(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.map(None)


@pytest.fixture
def test_mask(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mask(None, None, None, None, None, None)


@pytest.fixture
def test_max(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.max(None, None, None, None)


@pytest.fixture
def test_mean(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mean(None, None, None, None)


@pytest.fixture
def test_median(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.median(None, None, None, None)


@pytest.fixture
def test_memory_usage(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.memory_usage(None)


@pytest.fixture
def test_min(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.min(None, None, None, None)


@pytest.fixture
def test_mod(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mod(None, None, None)


@pytest.fixture
def test_mode(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mode()


@pytest.fixture
def test_mul(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.mul(None, None, None)


@pytest.fixture
def test_multiply(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.multiply(None, None, None)


@pytest.fixture
def test_name(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.name


@pytest.fixture
def test_nbytes(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.nbytes


@pytest.fixture
def test_ndim(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ndim


@pytest.fixture
def test_ne(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ne(None, None, None)


@pytest.fixture
def test_nlargest(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.nlargest(None)


@pytest.fixture
def test_nonzero(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.nonzero()


@pytest.fixture
def test_notnull(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.notnull()


@pytest.fixture
def test_nsmallest(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.nsmallest(None)


@pytest.fixture
def test_nunique(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.nunique(None)


@pytest.fixture
def test_pct_change(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.pct_change(None, None, None, None)


@pytest.fixture
def test_pipe(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.pipe(None, None)


@pytest.fixture
def test_plot(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.plot(None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None)


@pytest.fixture
def test_pop(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.pop(None)


@pytest.fixture
def test_pow(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.pow(None, None, None)


@pytest.fixture
def test_prod(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.prod(None, None, None, None)


@pytest.fixture
def test_product(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.product(None, None, None, None)


@pytest.fixture
def test_ptp(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ptp(None, None, None, None)


@pytest.fixture
def test_put(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.put(None)


@pytest.fixture
def test_quantile(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.quantile(None)


@pytest.fixture
def test_radd(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.radd(None, None, None)


@pytest.fixture
def test_rank(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rank(None, None, None, None, None)


@pytest.fixture
def test_ravel(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.ravel(None)


@pytest.fixture
def test_rdiv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rdiv(None, None, None)


@pytest.fixture
def test_real(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.real


@pytest.fixture
def test_reindex(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reindex(None)


@pytest.fixture
def test_reindex_axis(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reindex_axis(None, None)


@pytest.fixture
def test_reindex_like(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reindex_like(None, None, None, None)


@pytest.fixture
def test_rename(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rename(None)


@pytest.fixture
def test_rename_axis(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rename_axis(None, None, None)


@pytest.fixture
def test_reorder_levels(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reorder_levels(None)


@pytest.fixture
def test_repeat(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.repeat(None, None)


@pytest.fixture
def test_replace(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.replace(None, None, None, None, None, None)


@pytest.fixture
def test_resample(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.resample(None, None, None, None, None, None, None, None,
                            None, None, None, None)


@pytest.fixture
def test_reset_index(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reset_index(None, None, None)


@pytest.fixture
def test_reshape(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.reshape(None)


@pytest.fixture
def test_rfloordiv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rfloordiv(None, None, None)


@pytest.fixture
def test_rmod(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rmod(None, None, None)


@pytest.fixture
def test_rmul(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rmul(None, None, None)


@pytest.fixture
def test_rolling(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rolling(None, None, None, None, None, None, None)


@pytest.fixture
def test_round(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.round(None, None)


@pytest.fixture
def test_rpow(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rpow(None, None, None)


@pytest.fixture
def test_rsub(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rsub(None, None, None)


@pytest.fixture
def test_rtruediv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.rtruediv(None, None, None)


@pytest.fixture
def test_sample(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sample(None, None, None, None, None)


@pytest.fixture
def test_searchsorted(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.searchsorted(None, None)


@pytest.fixture
def test_select(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.select(None)


@pytest.fixture
def test_sem(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sem(None, None, None, None, None)


@pytest.fixture
def test_set_axis(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.set_axis(None, None)


@pytest.fixture
def test_set_value(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.set_value(None, None)


@pytest.fixture
def test_shape(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.shape


@pytest.fixture
def test_shift(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.shift(None, None)


@pytest.fixture
def test_size(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.size


@pytest.fixture
def test_skew(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.skew(None, None, None, None)


@pytest.fixture
def test_slice_shift(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.slice_shift(None)


@pytest.fixture
def test_sort_index(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sort_index(None, None, None, None, None, None)


@pytest.fixture
def test_sort_values(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sort_values(None, None, None, None)


@pytest.fixture
def test_sortlevel(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sortlevel(None, None)


@pytest.fixture
def test_squeeze(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.squeeze(None)


@pytest.fixture
def test_std(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.std(None, None, None, None, None)


@pytest.fixture
def test_strides(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.strides


@pytest.fixture
def test_sub(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sub(None, None, None)


@pytest.fixture
def test_subtract(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.subtract(None, None, None)


@pytest.fixture
def test_sum(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.sum(None, None, None, None)


@pytest.fixture
def test_swapaxes(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.swapaxes(None, None)


@pytest.fixture
def test_swaplevel(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.swaplevel(None, None)


@pytest.fixture
def test_tail(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.tail(None)


@pytest.fixture
def test_take(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.take(None, None, None, None)


@pytest.fixture
def test_to_clipboard(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_clipboard(None, None)


@pytest.fixture
def test_to_csv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_csv(None, None, None, None, None, None, None, None,
                          None, None)


@pytest.fixture
def test_to_dense(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_dense()


@pytest.fixture
def test_to_dict(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_dict()


@pytest.fixture
def test_to_excel(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_excel(None, None, None, None, None, None, None, None,
                            None, None, None, None, None, None)


@pytest.fixture
def test_to_frame(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_frame(None)


@pytest.fixture
def test_to_hdf(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_hdf(None, None)


@pytest.fixture
def test_to_json(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_json(None, None, None, None, None, None, None)


@pytest.fixture
def test_to_latex(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_latex(None, None, None, None, None, None, None, None,
                            None, None, None, None, None, None, None, None,
                            None, None)


@pytest.fixture
def test_to_msgpack(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_msgpack(None, None)


@pytest.fixture
def test_to_period(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_period(None)


@pytest.fixture
def test_to_pickle(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_pickle(None)


@pytest.fixture
def test_to_sparse(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_sparse(None)


@pytest.fixture
def test_to_sql(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_sql(None, None, None, None, None, None, None, None)


@pytest.fixture
def test_to_string(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_string(None, None, None, None, None, None, None, None)


@pytest.fixture
def test_to_timestamp(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_timestamp(None, None)


@pytest.fixture
def test_to_xarray(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.to_xarray()


@pytest.fixture
def test_tolist(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.tolist()


@pytest.fixture
def test_transform(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.transform(None, None)


@pytest.fixture
def test_transpose(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.transpose(None)


@pytest.fixture
def test_truediv(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.truediv(None, None, None)


@pytest.fixture
def test_truncate(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.truncate(None, None, None)


@pytest.fixture
def test_tshift(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.tshift(None, None)


@pytest.fixture
def test_tz_convert(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.tz_convert(None, None, None)


@pytest.fixture
def test_tz_localize(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.tz_localize(None, None, None, None)


@pytest.fixture
def test_unique(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.unique()


@pytest.fixture
def test_unstack(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.unstack(None)


@pytest.fixture
def test_update(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.update(None)


@pytest.fixture
def test_valid(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.valid(None)


@pytest.fixture
def test_value_counts(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.value_counts(None, None, None, None)


@pytest.fixture
def test_values(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.values


@pytest.fixture
def test_var(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.var(None, None, None, None, None)


@pytest.fixture
def test_view(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.view(None)


@pytest.fixture
def test_where(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.where(None, None, None, None, None, None)


@pytest.fixture
def test_xs(ray_series, pandas_series):
    with pytest.raises(NotImplementedError):
        ray_series.xs(None, None, None)
