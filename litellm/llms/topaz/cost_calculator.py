"""
Topaz Labs video enhancement cost.

NOL-519: Topaz logged $0 COGS on every restore, and unlike the other providers
this could not be fixed with a per-second rate. Topaz bills CREDITS for the
frames its engine processes, so cost tracks the source geometry, the output
geometry and the scale factor between them, non-monotonically: measured against
Topaz's own estimate endpoint, one classic-engine restore costs 0.25 credits
per 5s from a 960x720 source and 9.67 from a 7680x4320 source AT THE SAME 720p
OUTPUT TIER. No static per-second rate can straddle a ~39x spread.

So this does not model the cost function - it prices the number Topaz itself
reports. `estimates.cost` is a [lower, upper] credit range on the job's status
object, and Topaz bills the LOWER bound, which the video transform captures as
usage.topaz_credits. All that is left is credits -> USD.
"""

import litellm

# The Starter rate published at topazlabs.com/enhance-api. Deliberately the
# worst of the three plans (Developer $0.10, Scale $0.08) so a recorded cost
# holds as an upper bound on whichever plan the account is actually on - a COGS
# ledger that errs low is the failure this ticket exists to fix. Kept in
# lockstep with topaz.USDPerCredit in nolgia-api.
DEFAULT_USD_PER_CREDIT = 0.12


def cost_per_credit(model: str) -> float:
    """
    Resolve the USD-per-credit rate for a Topaz model from the cost map,
    falling back to the published Starter rate.

    The map is read directly rather than through get_model_info() because that
    helper raises for an unmapped model, and an unmapped Topaz engine must still
    price at the published rate rather than silently recording $0 - which is
    precisely the regression this file exists to prevent.
    """
    provider = litellm.LlmProviders.TOPAZ.value
    bare_model = model.split("/")[-1]
    for cost_key in (f"{provider}/{bare_model}", model, bare_model):
        cost_entry = litellm.model_cost.get(cost_key)
        if cost_entry:
            rate = cost_entry.get("output_cost_per_credit")
            if rate is not None:
                return float(rate)
            break
    return DEFAULT_USD_PER_CREDIT


def cost_calculator(model: str, topaz_credits: object) -> float:
    """
    USD for one Topaz enhancement, from the credits Topaz reported it will bill.

    Returns 0.0 when the credit count is missing or unusable. That is the same
    outcome as before this existed, and it is deliberate: a restore whose quote
    could not be read should record nothing rather than an invented figure, and
    the NOL-535 ledger guard is what catches a model that never records at all.
    """
    if isinstance(topaz_credits, bool) or not isinstance(topaz_credits, (int, float)):
        return 0.0
    if topaz_credits <= 0:
        return 0.0
    return float(topaz_credits) * cost_per_credit(model)
