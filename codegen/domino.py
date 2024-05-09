
allowed_states = ['d0', 'd1', 'd2', 'l2', 'l3', 'd4', 'd5', 'd6']

def emit_code(x, a, b):

    a_diff = x + b # b is in the neighbourhood of a
    b_diff = x + a # a is in the neighbourhood of b

    a_combs = [(('an%d' % i), ('a%s%d' % ('dl'[a], i + a_diff))) for i in range(4)]
    b_combs = [(('bn%d' % i), ('b%s%d' % ('dl'[b], i + b_diff))) for i in range(4)]

    a_combs = [t for t in a_combs if t[1][1:] in allowed_states]
    b_combs = [t for t in b_combs if t[1][1:] in allowed_states]

    if len(a_combs) == 0 or len(b_combs) == 0:
        return ''

    res = ['// x = %d, a = %d, b = %d' % (x, a, b)]
    res.append('uint32_t contradiction = x%d | (%s) | (%s);' % (
        x,
        ' & '.join(['(%s | %s)' % t for t in a_combs]),
        ' & '.join(['(%s | %s)' % t for t in b_combs])
    ))

    for l, r in a_combs + b_combs:
        res.append('g%s &= %s | contradiction;' % (l, r))
        res.append('g%s &= %s | contradiction;' % (r, l))

    return '\n    {\n%s    }' % ''.join(['        %s\n' % l for l in res])


def main():

    src = '''template<bool Vertical> _DI_ bool apply_domino_rule(%s) {

    constexpr bool Horizontal = !Vertical;

    // the inputs to this function specify which possibilities have
    // been ruled out (e.g. ad0 == 'we have ruled out dead with 0
    // live neighbours') so forced_live is the conjunction of all
    // of the ad? variables and forced_dead is the conjunction of
    // all of the al? variables:
    uint32_t forced_live = %s;
    uint32_t forced_dead = %s;

    // now we determine whether the cells immediately to the left
    // and right of the current cell are forced live/dead:
    uint32_t flla = kc::shift_plane<Horizontal,  1>(forced_live);
    uint32_t flra = kc::shift_plane<Horizontal, -1>(forced_live);
    uint32_t fdla = kc::shift_plane<Horizontal,  1>(forced_dead);
    uint32_t fdra = kc::shift_plane<Horizontal, -1>(forced_dead);

    // these variables are set to true if we have ruled out a
    // particular number of live cells amongst the cells marked
    // 'x' below:
    //
    // nnn
    // xax
    // xbx
    // nnn
    //
    // where we're in the frame of reference of the cell 'a':
    uint32_t x0, x1, x2, x3, x4;

    {
        // determine which of the x-cells in the row of 'b'
        // are forced to be live/dead:
        uint32_t fllb = kc::shift_plane<Vertical, -1>(flla);
        uint32_t flrb = kc::shift_plane<Vertical, -1>(flra);
        uint32_t fdlb = kc::shift_plane<Vertical, -1>(fdla);
        uint32_t fdrb = kc::shift_plane<Vertical, -1>(fdra);

        // apply a single full-adder to collapse the 4 variables
        // {flla, flra, fllb, flrb} into 3 variables {fl2, fl1, flrb}
        // with respective weights {2, 1, 1}, and do the same for
        // the forced-dead cells:
        uint32_t fl2 = apply_maj3(flla, flra, fllb);
        uint32_t fl1 = apply_xor3(flla, flra, fllb);
        uint32_t fd2 = apply_maj3(fdla, fdra, fdlb);
        uint32_t fd1 = apply_xor3(fdla, fdra, fdlb);

        // determine which counts of x-cells are excluded:
        x0 = (fl2 | (fl1 | flrb));
        x1 = (fl2 | (fl1 & flrb)) | (fd2 & (fd1 & fdrb));
        x2 = (fl2 & (fl1 | flrb)) | (fd2 & (fd1 | fdrb));
        x3 = (fl2 & (fl1 & flrb)) | (fd2 | (fd1 & fdrb));
        x4 =                        (fd2 | (fd1 | fdrb));
    }

    // these variables are set to true if we have ruled out a
    // particular number of live cells amongst the three 'n'
    // cells adjacent to 'a':
    uint32_t an0, an1, an2, an3;

    // and analogously for the 'n' cells adjacent to 'b':
    uint32_t bn0, bn1, bn2, bn3;

    // perform full-adders on the row 'xax':
    uint32_t fl2 = apply_maj3(flla, flra, forced_live);
    uint32_t fl1 = apply_xor3(flla, flra, forced_live);
    uint32_t fd2 = apply_maj3(fdla, fdra, forced_dead);
    uint32_t fd1 = apply_xor3(fdla, fdra, forced_dead);

    {
        // determine which counts of cells are excluded
        // in the row 'xax':
        uint32_t n0 = fl1;
        uint32_t n1 = fl2 | (fd2 & fd1);
        uint32_t n2 = fd2 | (fl2 & fl1);
        uint32_t n3 = fd1;

        // communicate vertically to get the excluded
        // counts in the two rows of 'n' cells:
        an0 = kc::shift_plane<Vertical,  1>(n0);
        an1 = kc::shift_plane<Vertical,  1>(n1);
        an2 = kc::shift_plane<Vertical,  1>(n2);
        an3 = kc::shift_plane<Vertical,  1>(n3);
        bn0 = kc::shift_plane<Vertical, -2>(n0);
        bn1 = kc::shift_plane<Vertical, -2>(n1);
        bn2 = kc::shift_plane<Vertical, -2>(n2);
        bn3 = kc::shift_plane<Vertical, -2>(n3);
    }

    // we deduce information about the neighbour counts of
    // a and b by initialising these variables to 'impossible'
    // and then doing a case-bash over possible configurations
    // of the six cells 'xaxxbx'. Although there are 2**6 = 64
    // combinations, we are invariant under permuting the 'x's
    // which reduces to 5 * 2 * 2 = 20 cases. Moreover, 4 of
    // these are incompatible with CGoL still-lifes so 16 cases
    // remain:
    uint32_t gan0 = 0xffffffffu;
    uint32_t gan1 = 0xffffffffu;
    uint32_t gan2 = 0xffffffffu;
    uint32_t gan3 = 0xffffffffu;
    uint32_t gbn0 = 0xffffffffu;
    uint32_t gbn1 = 0xffffffffu;
    uint32_t gbn2 = 0xffffffffu;
    uint32_t gbn3 = 0xffffffffu;
''' % (
        ', '.join(['uint32_t &a%s' % x for x in allowed_states]),
        ' & '.join(['a%s' % x for x in allowed_states if 'd' in x]),
        ' & '.join(['a%s' % x for x in allowed_states if 'l' in x])
    )

    for x in allowed_states:
        src += '''
    uint32_t b%s = kc::shift_plane<Vertical, -1>(a%s);
    uint32_t ga%s = 0xffffffffu;
    uint32_t gb%s = 0xffffffffu;
''' % (x, x, x, x)

    src += '\n    // now do the 16 cases in turn:'

    src += ''.join([emit_code(x, a, b) for x in range(5) for a in range(2) for b in range(2)]) + '\n\n'

    src += '''
    {
        // determine which combinations have been disproved:
        uint32_t gn0 = kc::shift_plane<Vertical, -1>(gan0) | kc::shift_plane<Vertical, 2>(gbn0);
        uint32_t gn1 = kc::shift_plane<Vertical, -1>(gan1) | kc::shift_plane<Vertical, 2>(gbn1);
        uint32_t gn2 = kc::shift_plane<Vertical, -1>(gan2) | kc::shift_plane<Vertical, 2>(gbn2);
        uint32_t gn3 = kc::shift_plane<Vertical, -1>(gan3) | kc::shift_plane<Vertical, 2>(gbn3);

        // determine which combinations have been proved:
        uint32_t gf0 = gn1 & gn2 & gn3;
        uint32_t gf1 = gn0 & gn2 & gn3;
        uint32_t gf2 = gn0 & gn1 & gn3;
        uint32_t gf3 = gn0 & gn1 & gn2;

        // determine whether to propagate live/dead to unknown neighbours:
        uint32_t pl = gf3 | (gf2 & fd1) | (gf1 & fd2);
        uint32_t pd = gf0 | (gf1 & fl2) | (gf2 & fl1);
        uint32_t turn_live = pl | kc::shift_plane<Horizontal, -1>(pl) | kc::shift_plane<Horizontal, 1>(pl);
        uint32_t turn_dead = pd | kc::shift_plane<Horizontal, -1>(pd) | kc::shift_plane<Horizontal, 1>(pd);

        // if unknown, set accordingly:
        forced_live |= (turn_live &~ forced_dead);
        forced_dead |= (turn_dead &~ forced_live);
    }

    // aggregate all of the information that we have learned into
    // the input variables in-place (they are references) whilst
    // tracking whether or not we have made any new deductions:
    uint32_t improvements = 0;
'''

    for x in allowed_states:
        y = {'l': 'forced_dead', 'd': 'forced_live'}[x[0]]
        src += '    ga%s |= %s | a%s | kc::shift_plane<Vertical, 1>(gb%s); improvements |= (ga%s &~ a%s); a%s = ga%s;\n' % (x, y, x, x, x, x, x, x)

    src += '''
    improvements = hh::ballot_32(improvements != 0);
    return (improvements != 0);
}
'''
    print(src)


if __name__ == '__main__':

    main()
