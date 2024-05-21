
allowed_states = ['d0', 'd1', 'd2', 'l2', 'l3', 'd4', 'd5', 'd6']

def emit_case(state):

    assert(state in allowed_states)

    stable_neighbours = int(state[1])

    fd, fl, us = {
        'd': ('not_high', 'not_stable', 'fl_next'),
        'l': ('not_stable', 'not_low', 'fd_next')
    }[state[0]]

    thresholds = [i - stable_neighbours for i in range(6, 9)]

    # these are the variables which constrain the current neighbour count:
    lt2, lt3, lt4 = ['ub%d' % i for i in thresholds]
    ge2, ge3, ge4 = ['lb%d' % i for i in thresholds]

    src = '''
    {
        uint32_t fd_next = (%s | (%s & %s) | %s) | a%s;
        uint32_t fl_next = (%s & (%s | %s) & %s) | a%s;
        unstable_%s = %s;
        g%s &= fd_next;
        g%s &= fl_next;
    }
''' % (
        lt2, lt3, fd, ge4, state,
        ge2, ge3, fl, lt4, state,
        state, us, fd, fl
    )

    return src


def main():

    src = '''_DI_ bool inplace_advance_unknown(
        %s,
        uint32_t &not_low, uint32_t &not_high, uint32_t &not_stable,
        uint32_t stator, int max_width = 28, int max_height = 28, uint32_t max_pop = 784, uint32_t* smem = nullptr
    ) {

    // obtain lower and upper bounds in binary:
    uint4 lb_bin = kc::sum16<false>(not_low, not_low & not_stable);
    uint4 ub_bin = kc::sum16<true>(not_high, not_high & not_stable);

    // lb[i] := "we know that (curr_pop - stable_pop >= i - 4)"
    uint32_t lb0 = lb_bin.w | lb_bin.z;
    uint32_t lb4 = lb_bin.w;
    uint32_t lb8 = lb_bin.w & lb_bin.z;
    uint32_t lb2 = lb4 | (lb0 & lb_bin.y);
    uint32_t lb6 = lb8 | (lb4 & lb_bin.y);
    uint32_t lb1 = lb2 | (lb0 & lb_bin.x);
    uint32_t lb3 = lb4 | (lb2 & lb_bin.x);
    uint32_t lb5 = lb6 | (lb4 & lb_bin.x);
    uint32_t lb7 = lb8 | (lb6 & lb_bin.x);

    // ub[i] := "we know that (curr_pop - stable_pop  < i - 4)"
    uint32_t ub8 = ub_bin.w | ub_bin.z;
    uint32_t ub4 = ub_bin.w;
    uint32_t ub0 = ub_bin.w & ub_bin.z;
    uint32_t ub6 = ub4 | (ub8 & ub_bin.y);
    uint32_t ub2 = ub0 | (ub4 & ub_bin.y);
    uint32_t ub7 = ub6 | (ub8 & ub_bin.x);
    uint32_t ub5 = ub4 | (ub6 & ub_bin.x);
    uint32_t ub3 = ub2 | (ub4 & ub_bin.x);
    uint32_t ub1 = ub0 | (ub2 & ub_bin.x);

    uint32_t gnot_low    = 0xffffffffu;
    uint32_t gnot_high   = 0xffffffffu;
    uint32_t gnot_stable = 0xffffffffu;
''' % ', '.join(['uint32_t &a%s' % x for x in allowed_states])

    src += '    uint32_t %s;\n' % ', '.join(['unstable_%s' % x for x in allowed_states])

    for state in allowed_states:
        src += emit_case(state)

    src += '''
    uint32_t forced_stable = kc::get_forced_stable(gnot_stable, ad0, stator, max_width, max_height, max_pop);
    gnot_low |= forced_stable;
    gnot_high |= forced_stable;
    uint32_t improvements = 0;
'''

    for state in allowed_states:
        src += '    uint32_t ga%s = a%s | (forced_stable & unstable_%s); improvements |= (ga%s &~ a%s); a%s = ga%s;\n' % ((state,)*7)

    src += '''
    // store in smem if appropriate:
    if (smem != nullptr) {
        smem[threadIdx.x] = unstable_d0;
        smem[threadIdx.x + 32] = unstable_d1;
        smem[threadIdx.x + 64] = unstable_d2;
        smem[threadIdx.x + 96] = unstable_l2;
        smem[threadIdx.x + 128] = unstable_l3;
        smem[threadIdx.x + 160] = unstable_d4;
        smem[threadIdx.x + 192] = unstable_d5;
        smem[threadIdx.x + 224] = unstable_d6;
    }

    // advance by one generation in-place:
    not_low = gnot_low;
    not_high = gnot_high;
    not_stable = gnot_stable;

    // have we deduced new information about the stable state?
    improvements = hh::ballot_32(improvements != 0);
    return (improvements != 0);
}
'''

    print(src)


if __name__ == '__main__':

    main()

