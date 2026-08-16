/*
 * position.h — Board representation, magic bitboards, Zobrist hashing,
 *              move generation (pseudo-legal + legal filter), do/undo move.
 *
 * Design choices:
 *   • All magic tables are global statics (2.4 MB total) — initialised once
 *     by Position::init() called from main().
 *   • Board state history is a fixed-size inline array (no heap allocations).
 *   • gen_moves() returns pseudo-legal moves; gen_legal() runs the legal
 *     filter in-place, keeping only moves that don't leave the king in check.
 */

#pragma once
#include "types.h"

#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <cctype>
#include <cstdlib>

// ─────────────────────────── magic table storage ───────────────────────────

struct Magic {
    Bitboard  mask;
    Bitboard  magic;
    Bitboard *attacks;
    int       shift;

    Bitboard operator()(Bitboard occ) const {
        return attacks[((occ & mask) * magic) >> shift];
    }
};

// Pre-computed magic numbers — validated collision-free at init time.
// Rook:   variable shift (64 − popcount(mask)), at most 12 relevant bits → 4096 entries.
// Bishop: variable shift,                       at most  9 relevant bits →  512 entries.
static Magic   g_rook_magic[64];
static Magic   g_bishop_magic[64];
static Bitboard g_rook_attacks_table[64][4096];
static Bitboard g_bishop_attacks_table[64][512];
static Bitboard g_knight_attacks[64];
static Bitboard g_king_attacks[64];
static Bitboard g_pawn_attacks[2][64];  // [color][sq]

inline Bitboard rook_attacks  (Square sq, Bitboard occ) { return g_rook_magic[sq](occ); }
inline Bitboard bishop_attacks(Square sq, Bitboard occ) { return g_bishop_magic[sq](occ); }
inline Bitboard queen_attacks (Square sq, Bitboard occ) { return rook_attacks(sq,occ)|bishop_attacks(sq,occ); }

// ─────────────────────────── known good magics ─────────────────────────────

static const Bitboard ROOK_MAGICS[64] = {
    0x2080001620400080ULL, 0x0440029000406004ULL, 0x0080200010008008ULL, 0x0100082010000502ULL,
    0x0480040002800800ULL, 0x8880140012008001ULL, 0x42002C0588020029ULL, 0x8200022213048044ULL,
    0x0409800180C00020ULL, 0x0012002041020094ULL, 0x5002801000A00080ULL, 0x8802002012000B40ULL,
    0x0000800400800800ULL, 0x0002808004000200ULL, 0x2401010401000200ULL, 0x08C0802080004100ULL,
    0x0080114001406000ULL, 0x00D0004000200048ULL, 0x0000410010200109ULL, 0x0008008008100080ULL,
    0x4004008008000481ULL, 0x8104008002008004ULL, 0x4000040008820150ULL, 0x0000020000440081ULL,
    0x1000800080204002ULL, 0x0040008100310040ULL, 0x0208104200220080ULL, 0x2100100100210008ULL,
    0x2280110100080004ULL, 0x8000020080040080ULL, 0x9800888400100102ULL, 0x0000802080004100ULL,
    0x2020004000808000ULL, 0x0823C02002401000ULL, 0x1611004011002001ULL, 0x0000081001002100ULL,
    0x1010080080800400ULL, 0x8000040080800200ULL, 0x0300100204008801ULL, 0x2812050486000044ULL,
    0x2000814000218005ULL, 0x0410002000404000ULL, 0x3810004020010100ULL, 0x2001001000210008ULL,
    0x2048000400088080ULL, 0x8000040002008080ULL, 0x0800020001008080ULL, 0x0180208400520021ULL,
    0x0080448001002300ULL, 0x4204804001002500ULL, 0x014120118A420200ULL, 0x0100080080100080ULL,
    0x9008000804008080ULL, 0x1000040080020080ULL, 0x1284280230010400ULL, 0x0804110084204200ULL,
    0x0042214100508001ULL, 0x0300204000108101ULL, 0x0100200100100C41ULL, 0x0420852100100009ULL,
    0x008200A41008204AULL, 0x0282000448011062ULL, 0x0008008228100144ULL, 0x0A000C0085005022ULL,
};

static const Bitboard BISHOP_MAGICS[64] = {
    0x0018911026004900ULL, 0x0820020082248000ULL, 0x0442043040800800ULL, 0x0084404484000000ULL,
    0x0184042000840C10ULL, 0x0001010840082470ULL, 0x001884C820106008ULL, 0x8010108815082000ULL,
    0x0000040808212400ULL, 0x8800080220840100ULL, 0x0000240845810800ULL, 0x0400022A02008520ULL,
    0x0200020211004009ULL, 0x200A408220200002ULL, 0x0000004108A01080ULL, 0x1000082406185404ULL,
    0x2108804110412214ULL, 0x4010990204154400ULL, 0x0081021204010A00ULL, 0x0344041824001020ULL,
    0x110C800400A01000ULL, 0x0101080200822010ULL, 0x0002028051142040ULL, 0x0090903024040200ULL,
    0x0410284110329001ULL, 0x0210080050810100ULL, 0x1402110008014400ULL, 0x0060080002081010ULL,
    0x0380840002020201ULL, 0x580202008148060AULL, 0x080440C413080640ULL, 0x000A00281A090102ULL,
    0x001014240260089AULL, 0x902828040042C420ULL, 0x0300180400C20400ULL, 0x0089E08400080210ULL,
    0x2028020400001010ULL, 0x2020808602210100ULL, 0x0468080045092100ULL, 0x0004010048002400ULL,
    0x00886A0220005000ULL, 0x0081009004501000ULL, 0x001E010402100100ULL, 0x8040122018012108ULL,
    0x0000080104000841ULL, 0x04440804880A2100ULL, 0x40082200A2200400ULL, 0x0001010408840108ULL,
    0x0081040120092000ULL, 0x4709128A10060080ULL, 0x0087310401310020ULL, 0x4040080084040000ULL,
    0x0202419021024408ULL, 0x2040418408008804ULL, 0x002920244C820802ULL, 0x00A0046080810054ULL,
    0x2002008421111005ULL, 0x0029020100821020ULL, 0x1102001040441004ULL, 0x38A0822082420220ULL,
    0x1001000088102400ULL, 0x0082029242104100ULL, 0x0081098890041040ULL, 0x60092104058C0500ULL,
};

// ─────────────────────────── board state ───────────────────────────────────

// Per-move state that must be restored on undo_move
struct StateInfo {
    uint64_t  hash;
    int       ep_square;       // -1 = none
    int       castle_rights;   // bitmask: CR_WK | CR_WQ | CR_BK | CR_BQ
    int       halfmove_clock;
    PieceType captured_piece;  // piece type captured (NO_PIECE = quiet)
};

static constexpr int CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8;

// ─────────────────────────── Position ──────────────────────────────────────

struct Position {

    // ── Board arrays ───────────────────────────────────────────────────────
    Bitboard  bb[2][6];        // [color][piece_type] bitboard
    Bitboard  occupied[2];     // all squares occupied by each color
    Bitboard  all;             // union of both occupied[]
    PieceType piece_on[64];    // piece type on each square (NO_PIECE = empty)
    Color     color_on[64];    // color of piece on each square
    int       king_sq[2];      // king square for each color
    Color     stm;             // side to move
    int       piece_count;     // non-king piece count (for NNUE bucket)

    // History stack: large enough for max-ply search + game moves
    static constexpr int HIST_MAX = MAX_PLY + 512;
    StateInfo history[HIST_MAX];
    int       history_top;     // next free slot index
    StateInfo cur;             // active state (copied into history on do_move)

    // ── One-time global initialisation ─────────────────────────────────────
    // Call once from main() before creating any Position.
    static void init();

    // ── Setup ──────────────────────────────────────────────────────────────

    void clear() {
        memset(bb, 0, sizeof bb);
        memset(occupied, 0, sizeof occupied);
        all = 0;
        for (int i = 0; i < 64; ++i) { piece_on[i] = NO_PIECE; color_on[i] = WHITE; }
        king_sq[WHITE] = king_sq[BLACK] = 0;
        piece_count = 0;
        stm = WHITE;
        cur = {0, -1, 0, 0, NO_PIECE};
        history_top = 0;
    }

    void set_from_fen(const std::string &fen);

    // ── Piece manipulation (update Zobrist hash incrementally) ─────────────

    void place(Color c, PieceType pt, Square s);
    void remove(Color c, PieceType pt, Square s);

    // ── Attack queries ──────────────────────────────────────────────────────

    bool sq_attacked(Square sq, Color by) const {
        if (g_pawn_attacks[!by][sq]  & bb[by][PAWN])   return true;
        if (g_knight_attacks[sq]     & bb[by][KNIGHT])  return true;
        if (g_king_attacks[sq]       & bb[by][KING])    return true;
        Bitboard occ = all;
        if (bishop_attacks(sq, occ)  & (bb[by][BISHOP]|bb[by][QUEEN])) return true;
        if (rook_attacks(sq, occ)    & (bb[by][ROOK]  |bb[by][QUEEN])) return true;
        return false;
    }

    bool in_check() const { return sq_attacked(king_sq[stm], static_cast<Color>(!stm)); }

    // ── Move making ─────────────────────────────────────────────────────────

    void do_move(Move m);
    void undo_move(Move m);

    // ── Move generation ─────────────────────────────────────────────────────
    //
    // gen_pseudo()   writes pseudo-legal moves to list, returns count.
    // gen_legal()    filters gen_pseudo() output, returns legal count.
    //
    // captures_only=true skips quiet moves (used in qsearch).

    int gen_pseudo(Move *list, bool captures_only = false) const;
    int gen_legal (Move *list, bool captures_only = false);

    // Single-call legal movegen (convenience wrapper)
    int gen_moves(Move *list, bool captures_only = false) {
        return gen_legal(list, captures_only);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    // Number of non-king pieces (for NNUE material bucket)
    int material_count() const { return piece_count; }

    // Detect draw conditions
    bool is_insufficient_material() const;
    bool is_draw() const;
    bool is_repetition(int root_distance) const;

    // UCI move parsing / printing
    Move        parse_uci(const std::string &s) const;
    std::string move_uci(Move m) const;
};

// ─────────────────────────── Zobrist ───────────────────────────────────────

static uint64_t g_zobrist_piece[2][6][64];
static uint64_t g_zobrist_ep[64];
static uint64_t g_zobrist_castle[16];
static uint64_t g_zobrist_stm;

// ─────────────────────────── Position::init() ──────────────────────────────

inline void Position::init() {

    // ── Zobrist keys ───────────────────────────────────────────────────────
    {
        uint64_t s = 0xDEADBEEFCAFE1234ULL;
        auto next = [&]() -> uint64_t {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            return s * 0x2545F4914F6CDD1DULL;
        };
        for (int c = 0; c < 2; c++)
            for (int p = 0; p < 6; p++)
                for (int sq = 0; sq < 64; sq++)
                    g_zobrist_piece[c][p][sq] = next();
        for (int sq = 0; sq < 64; sq++) g_zobrist_ep[sq]     = next();
        for (int i  = 0; i  < 16; i++)  g_zobrist_castle[i]  = next();
        g_zobrist_stm = next();
    }

    // ── Slider attack helper (for magic initialisation) ────────────────────
    static const int ROOK_DELTAS[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int BISHOP_DELTAS[4][2] = {{1,1},{-1,1},{1,-1},{-1,-1}};

    auto sliding_attacks = [](Square sq, Bitboard occ,
                               const int deltas[][2], int ndelta) -> Bitboard {
        Bitboard ret = 0;
        int r = sq_rank(sq), f = sq_file(sq);
        for (int d = 0; d < ndelta; d++) {
            int dr = deltas[d][0], df = deltas[d][1];
            for (int nr = r+dr, nf = f+df;
                 nr >= 0 && nr < 8 && nf >= 0 && nf < 8;
                 nr += dr, nf += df) {
                Bitboard ns = sq_bb(make_sq(nf, nr));
                ret |= ns;
                if (occ & ns) break;
            }
        }
        return ret;
    };

    auto rook_mask = [](Square sq) -> Bitboard {
        Bitboard r = 0; int rank = sq_rank(sq), file = sq_file(sq);
        for (int nr = rank+1; nr <= 6; nr++) r |= sq_bb(make_sq(file, nr));
        for (int nr = rank-1; nr >= 1; nr--) r |= sq_bb(make_sq(file, nr));
        for (int nf = file+1; nf <= 6; nf++) r |= sq_bb(make_sq(nf, rank));
        for (int nf = file-1; nf >= 1; nf--) r |= sq_bb(make_sq(nf, rank));
        return r;
    };
    auto bishop_mask = [](Square sq) -> Bitboard {
        Bitboard r = 0; int rank = sq_rank(sq), file = sq_file(sq);
        for (int nr=rank+1,nf=file+1; nr<=6&&nf<=6; nr++,nf++) r |= sq_bb(make_sq(nf,nr));
        for (int nr=rank+1,nf=file-1; nr<=6&&nf>=1; nr++,nf--) r |= sq_bb(make_sq(nf,nr));
        for (int nr=rank-1,nf=file+1; nr>=1&&nf<=6; nr--,nf++) r |= sq_bb(make_sq(nf,nr));
        for (int nr=rank-1,nf=file-1; nr>=1&&nf>=1; nr--,nf--) r |= sq_bb(make_sq(nf,nr));
        return r;
    };

    // ── Non-slider attacks ─────────────────────────────────────────────────
    for (Square sq = 0; sq < 64; sq++) {
        int r = sq_rank(sq), f = sq_file(sq);
        // Knight
        Bitboard kn = 0;
        static const int KND[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
        for (auto &d : KND) {
            int nr = r+d[0], nf = f+d[1];
            if (nr>=0&&nr<8&&nf>=0&&nf<8) kn |= sq_bb(make_sq(nf,nr));
        }
        g_knight_attacks[sq] = kn;
        // King
        Bitboard kg = 0;
        for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++) {
            if (!dr && !df) continue;
            int nr = r+dr, nf = f+df;
            if (nr>=0&&nr<8&&nf>=0&&nf<8) kg |= sq_bb(make_sq(nf,nr));
        }
        g_king_attacks[sq] = kg;
        // Pawns
        g_pawn_attacks[WHITE][sq] = 0;
        if (f>0 && r<7) g_pawn_attacks[WHITE][sq] |= sq_bb(make_sq(f-1,r+1));
        if (f<7 && r<7) g_pawn_attacks[WHITE][sq] |= sq_bb(make_sq(f+1,r+1));
        g_pawn_attacks[BLACK][sq] = 0;
        if (f>0 && r>0) g_pawn_attacks[BLACK][sq] |= sq_bb(make_sq(f-1,r-1));
        if (f<7 && r>0) g_pawn_attacks[BLACK][sq] |= sq_bb(make_sq(f+1,r-1));
    }

    // ── Magic tables ───────────────────────────────────────────────────────
    for (Square sq = 0; sq < 64; sq++) {
        // Rook
        {
            Magic &m = g_rook_magic[sq];
            m.mask    = rook_mask(sq);
            m.magic   = ROOK_MAGICS[sq];
            m.shift   = 64 - popcount(m.mask);
            m.attacks = g_rook_attacks_table[sq];
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                m.attacks[idx] = sliding_attacks(sq, occ, ROOK_DELTAS, 4);
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
        // Bishop
        {
            Magic &m = g_bishop_magic[sq];
            m.mask    = bishop_mask(sq);
            m.magic   = BISHOP_MAGICS[sq];
            m.shift   = 64 - popcount(m.mask);
            m.attacks = g_bishop_attacks_table[sq];
            Bitboard occ = 0;
            do {
                int idx = (int)(((occ & m.mask) * m.magic) >> m.shift);
                m.attacks[idx] = sliding_attacks(sq, occ, BISHOP_DELTAS, 4);
                occ = (occ - 1) & m.mask;
            } while (occ);
        }
    }
}

// ─────────────────────────── Position method bodies ────────────────────────

inline void Position::place(Color c, PieceType pt, Square s) {
    bb[c][pt] |= sq_bb(s);
    occupied[c] |= sq_bb(s);
    all |= sq_bb(s);
    piece_on[s] = pt;
    color_on[s] = c;
    if (pt == KING) king_sq[c] = s;
    else piece_count++;
    cur.hash ^= g_zobrist_piece[c][pt][s];
}

inline void Position::remove(Color c, PieceType pt, Square s) {
    bb[c][pt] &= ~sq_bb(s);
    occupied[c] &= ~sq_bb(s);
    all &= ~sq_bb(s);
    piece_on[s] = NO_PIECE;
    color_on[s] = WHITE;
    if (pt != KING) piece_count--;
    cur.hash ^= g_zobrist_piece[c][pt][s];
}

inline void Position::set_from_fen(const std::string &fen) {
    clear();
    std::istringstream ss(fen);
    std::string board_part, stm_str, castle_str, ep_str;
    int halfmove = 0, fullmove = 1;
    ss >> board_part >> stm_str >> castle_str >> ep_str >> halfmove >> fullmove;

    int rank = 7, file = 0;
    for (char ch : board_part) {
        if (ch == '/') { rank--; file = 0; }
        else if (ch >= '1' && ch <= '8') { file += ch - '0'; }
        else {
            Color col = isupper(ch) ? WHITE : BLACK;
            PieceType pt = NO_PIECE;
            switch (tolower(ch)) {
                case 'p': pt=PAWN;   break; case 'n': pt=KNIGHT; break;
                case 'b': pt=BISHOP; break; case 'r': pt=ROOK;   break;
                case 'q': pt=QUEEN;  break; case 'k': pt=KING;   break;
            }
            if (pt != NO_PIECE) place(col, pt, make_sq(file, rank));
            file++;
        }
    }

    stm = (stm_str == "w") ? WHITE : BLACK;
    if (stm == BLACK) cur.hash ^= g_zobrist_stm;

    cur.castle_rights = 0;
    for (char ch : castle_str) {
        if (ch=='K') cur.castle_rights |= CR_WK;
        if (ch=='Q') cur.castle_rights |= CR_WQ;
        if (ch=='k') cur.castle_rights |= CR_BK;
        if (ch=='q') cur.castle_rights |= CR_BQ;
    }
    cur.hash ^= g_zobrist_castle[cur.castle_rights];

    cur.ep_square = -1;
    if (ep_str != "-") {
        cur.ep_square = make_sq(ep_str[0]-'a', ep_str[1]-'1');
        cur.hash ^= g_zobrist_ep[cur.ep_square];
    }
    cur.halfmove_clock = halfmove;
    cur.captured_piece = NO_PIECE;
}

inline void Position::do_move(Move m) {
    history[history_top++] = cur;

    cur.hash ^= g_zobrist_castle[cur.castle_rights];
    if (cur.ep_square >= 0) {
        cur.hash ^= g_zobrist_ep[cur.ep_square];
        cur.ep_square = -1;
    }

    // Null move (used in NMP)
    if (m == NULL_MOVE) {
        stm = static_cast<Color>(!stm);
        cur.hash ^= g_zobrist_stm;
        cur.captured_piece = NO_PIECE;
        cur.hash ^= g_zobrist_castle[cur.castle_rights];
        return;
    }

    const Square    from  = move_from(m);
    const Square    to    = move_to(m);
    const PieceType pt    = piece_on[from];
    const Color     us    = stm;
    const Color     them  = static_cast<Color>(!us);
    const MoveFlag  flags = move_flags(m);
    const PieceType promo = move_promo(m);

    cur.captured_piece = NO_PIECE;

    // Capture / en-passant removal
    if (flags == MF_EP) {
        Square cap_sq = (us == WHITE) ? to - 8 : to + 8;
        cur.captured_piece = PAWN;
        remove(them, PAWN, cap_sq);
    } else if (piece_on[to] != NO_PIECE) {
        cur.captured_piece = piece_on[to];
        remove(them, cur.captured_piece, to);
    }

    // Move the piece (promote if needed)
    remove(us, pt, from);
    PieceType landing = (flags == MF_PROMO) ? promo : pt;
    place(us, landing, to);

    // Rook shuffle for castling
    if (flags == MF_CASTLE) {
        if (to == 6)  { remove(us,ROOK,7);  place(us,ROOK,5);  }  // White K-side
        if (to == 2)  { remove(us,ROOK,0);  place(us,ROOK,3);  }  // White Q-side
        if (to == 62) { remove(us,ROOK,63); place(us,ROOK,61); }  // Black K-side
        if (to == 58) { remove(us,ROOK,56); place(us,ROOK,59); }  // Black Q-side
    }

    // Double pawn push → set en-passant square
    if (pt == PAWN && std::abs(to - from) == 16) {
        cur.ep_square = (us == WHITE) ? from + 8 : from - 8;
        cur.hash ^= g_zobrist_ep[cur.ep_square];
    }

    // Update castling rights (mask away rights for moved rook/king squares)
    static const int CASTLE_MASK[64] = {
        ~CR_WQ,15,15,15,~(CR_WK|CR_WQ),15,15,~CR_WK,
        15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
        ~CR_BQ,15,15,15,~(CR_BK|CR_BQ),15,15,~CR_BK
    };
    cur.castle_rights &= CASTLE_MASK[from] & CASTLE_MASK[to];
    cur.hash ^= g_zobrist_castle[cur.castle_rights];

    cur.halfmove_clock = (pt == PAWN || cur.captured_piece != NO_PIECE)
                         ? 0 : cur.halfmove_clock + 1;

    stm = them;
    cur.hash ^= g_zobrist_stm;
}

inline void Position::undo_move(Move m) {
    stm = static_cast<Color>(!stm);
    Color us   = stm;
    Color them = static_cast<Color>(!us);

    if (m != NULL_MOVE) {
        const Square   from     = move_from(m);
        const Square   to       = move_to(m);
        const MoveFlag flags    = move_flags(m);
        const PieceType promo   = move_promo(m);
        const PieceType landing = (flags == MF_PROMO) ? promo : piece_on[to];
        const PieceType captured = cur.captured_piece;  // read BEFORE restoring cur

        remove(us, landing, to);
        PieceType restore = (flags == MF_PROMO) ? PAWN : landing;
        place(us, restore, from);
        if (restore == KING) king_sq[us] = from;

        if (flags == MF_CASTLE) {
            if (to == 6)  { remove(us,ROOK,5);  place(us,ROOK,7);  }
            if (to == 2)  { remove(us,ROOK,3);  place(us,ROOK,0);  }
            if (to == 62) { remove(us,ROOK,61); place(us,ROOK,63); }
            if (to == 58) { remove(us,ROOK,59); place(us,ROOK,56); }
        }

        if (flags == MF_EP) {
            Square cap_sq = (us == WHITE) ? to - 8 : to + 8;
            place(them, PAWN, cap_sq);
        } else if (captured != NO_PIECE) {
            place(them, captured, to);
        }
    }

    cur = history[--history_top];
}

// ─────────────────────────── Move generation ───────────────────────────────

inline int Position::gen_pseudo(Move *list, bool captures_only) const {
    int n = 0;
    Color us    = stm;
    Color them  = static_cast<Color>(!us);
    Bitboard occ    = all;
    Bitboard mine   = occupied[us];
    Bitboard theirs = occupied[them];
    Bitboard targets = captures_only ? theirs : ~mine;

    // ── Pawns ───────────────────────────────────────────────────────────────
    {
        Bitboard pawns = bb[us][PAWN];
        if (us == WHITE) {
            Bitboard push1     = (pawns << 8) & ~occ;
            Bitboard push2     = ((push1 & RANK_3) << 8) & ~occ;
            Bitboard promo_p   = push1 & RANK_8; push1 &= ~RANK_8;
            Bitboard cap_r     = ((pawns & ~FILE_H) << 9) & theirs;
            Bitboard cap_l     = ((pawns & ~FILE_A) << 7) & theirs;
            Bitboard promo_r   = cap_r & RANK_8; cap_r &= ~RANK_8;
            Bitboard promo_l   = cap_l & RANK_8; cap_l &= ~RANK_8;

            if (!captures_only) {
                while (push1) { Square t=pop_lsb(push1); list[n++]=make_move(t-8,t); }
                while (push2) { Square t=pop_lsb(push2); list[n++]=make_move(t-16,t); }
            }
            while (cap_r)   { Square t=pop_lsb(cap_r);   list[n++]=make_move(t-9,t); }
            while (cap_l)   { Square t=pop_lsb(cap_l);   list[n++]=make_move(t-7,t); }
            // Promotions — explicit loops to avoid UB from pointer-comparing std::ref temporaries
            while (promo_p) { Square t=pop_lsb(promo_p); list[n++]=make_move(t-8,t,QUEEN,MF_PROMO); list[n++]=make_move(t-8,t,ROOK,MF_PROMO); list[n++]=make_move(t-8,t,BISHOP,MF_PROMO); list[n++]=make_move(t-8,t,KNIGHT,MF_PROMO); }
            while (promo_r) { Square t=pop_lsb(promo_r); list[n++]=make_move(t-9,t,QUEEN,MF_PROMO); list[n++]=make_move(t-9,t,ROOK,MF_PROMO); list[n++]=make_move(t-9,t,BISHOP,MF_PROMO); list[n++]=make_move(t-9,t,KNIGHT,MF_PROMO); }
            while (promo_l) { Square t=pop_lsb(promo_l); list[n++]=make_move(t-7,t,QUEEN,MF_PROMO); list[n++]=make_move(t-7,t,ROOK,MF_PROMO); list[n++]=make_move(t-7,t,BISHOP,MF_PROMO); list[n++]=make_move(t-7,t,KNIGHT,MF_PROMO); }
            // En-passant
            if (cur.ep_square >= 0) {
                Bitboard ep = sq_bb(cur.ep_square);
                if (pawns & ((ep >> 9) & ~FILE_H)) list[n++]=make_move(cur.ep_square-9, cur.ep_square, NO_PIECE, MF_EP);
                if (pawns & ((ep >> 7) & ~FILE_A)) list[n++]=make_move(cur.ep_square-7, cur.ep_square, NO_PIECE, MF_EP);
            }
        } else {
            Bitboard push1     = (pawns >> 8) & ~occ;
            Bitboard push2     = ((push1 & RANK_6) >> 8) & ~occ;
            Bitboard promo_p   = push1 & RANK_1; push1 &= ~RANK_1;
            Bitboard cap_r     = ((pawns & ~FILE_A) >> 9) & theirs;
            Bitboard cap_l     = ((pawns & ~FILE_H) >> 7) & theirs;
            Bitboard promo_r   = cap_r & RANK_1; cap_r &= ~RANK_1;
            Bitboard promo_l   = cap_l & RANK_1; cap_l &= ~RANK_1;

            if (!captures_only) {
                while (push1) { Square t=pop_lsb(push1); list[n++]=make_move(t+8,t); }
                while (push2) { Square t=pop_lsb(push2); list[n++]=make_move(t+16,t); }
            }
            while (cap_r)   { Square t=pop_lsb(cap_r);   list[n++]=make_move(t+9,t); }
            while (cap_l)   { Square t=pop_lsb(cap_l);   list[n++]=make_move(t+7,t); }
            // Promotions — explicit loops to avoid UB from pointer-comparing std::ref temporaries
            while (promo_p) { Square t=pop_lsb(promo_p); list[n++]=make_move(t+8,t,QUEEN,MF_PROMO); list[n++]=make_move(t+8,t,ROOK,MF_PROMO); list[n++]=make_move(t+8,t,BISHOP,MF_PROMO); list[n++]=make_move(t+8,t,KNIGHT,MF_PROMO); }
            while (promo_r) { Square t=pop_lsb(promo_r); list[n++]=make_move(t+9,t,QUEEN,MF_PROMO); list[n++]=make_move(t+9,t,ROOK,MF_PROMO); list[n++]=make_move(t+9,t,BISHOP,MF_PROMO); list[n++]=make_move(t+9,t,KNIGHT,MF_PROMO); }
            while (promo_l) { Square t=pop_lsb(promo_l); list[n++]=make_move(t+7,t,QUEEN,MF_PROMO); list[n++]=make_move(t+7,t,ROOK,MF_PROMO); list[n++]=make_move(t+7,t,BISHOP,MF_PROMO); list[n++]=make_move(t+7,t,KNIGHT,MF_PROMO); }
            if (cur.ep_square >= 0) {
                Bitboard ep = sq_bb(cur.ep_square);
                if (pawns & ((ep << 9) & ~FILE_A)) list[n++]=make_move(cur.ep_square+9, cur.ep_square, NO_PIECE, MF_EP);
                if (pawns & ((ep << 7) & ~FILE_H)) list[n++]=make_move(cur.ep_square+7, cur.ep_square, NO_PIECE, MF_EP);
            }
        }
    }

    // ── Knights ─────────────────────────────────────────────────────────────
    {
        Bitboard knights = bb[us][KNIGHT];
        while (knights) {
            Square f = pop_lsb(knights);
            Bitboard att = g_knight_attacks[f] & targets;
            while (att) { Square t=pop_lsb(att); list[n++]=make_move(f,t); }
        }
    }

    // ── Bishops ─────────────────────────────────────────────────────────────
    {
        Bitboard bishops = bb[us][BISHOP];
        while (bishops) {
            Square f = pop_lsb(bishops);
            Bitboard att = bishop_attacks(f, occ) & targets;
            while (att) { Square t=pop_lsb(att); list[n++]=make_move(f,t); }
        }
    }

    // ── Rooks ───────────────────────────────────────────────────────────────
    {
        Bitboard rooks = bb[us][ROOK];
        while (rooks) {
            Square f = pop_lsb(rooks);
            Bitboard att = rook_attacks(f, occ) & targets;
            while (att) { Square t=pop_lsb(att); list[n++]=make_move(f,t); }
        }
    }

    // ── Queens ──────────────────────────────────────────────────────────────
    {
        Bitboard queens = bb[us][QUEEN];
        while (queens) {
            Square f = pop_lsb(queens);
            Bitboard att = queen_attacks(f, occ) & targets;
            while (att) { Square t=pop_lsb(att); list[n++]=make_move(f,t); }
        }
    }

    // ── King ────────────────────────────────────────────────────────────────
    {
        Square ksq = king_sq[us];
        Bitboard att = g_king_attacks[ksq] & targets;
        while (att) { Square t=pop_lsb(att); list[n++]=make_move(ksq,t); }

        // Castling (always checked, never when captures_only since it's a quiet)
        if (!captures_only) {
            if (us == WHITE) {
                if ((cur.castle_rights & CR_WK) &&
                    !(all & 0x60ULL) &&
                    !sq_attacked(4,BLACK) && !sq_attacked(5,BLACK) && !sq_attacked(6,BLACK))
                    list[n++] = make_move(4, 6, NO_PIECE, MF_CASTLE);
                if ((cur.castle_rights & CR_WQ) &&
                    !(all & 0x0EULL) &&
                    !sq_attacked(4,BLACK) && !sq_attacked(3,BLACK) && !sq_attacked(2,BLACK))
                    list[n++] = make_move(4, 2, NO_PIECE, MF_CASTLE);
            } else {
                if ((cur.castle_rights & CR_BK) &&
                    !(all & 0x6000000000000000ULL) &&
                    !sq_attacked(60,WHITE) && !sq_attacked(61,WHITE) && !sq_attacked(62,WHITE))
                    list[n++] = make_move(60, 62, NO_PIECE, MF_CASTLE);
                if ((cur.castle_rights & CR_BQ) &&
                    !(all & 0x0E00000000000000ULL) &&
                    !sq_attacked(60,WHITE) && !sq_attacked(59,WHITE) && !sq_attacked(58,WHITE))
                    list[n++] = make_move(60, 58, NO_PIECE, MF_CASTLE);
            }
        }
    }

    return n;
}

inline int Position::gen_legal(Move *list, bool captures_only) {
    Move pseudo[MAX_MOVES];
    int np = gen_pseudo(pseudo, captures_only);
    Color us   = stm;
    Color them = static_cast<Color>(!us);
    int legal  = 0;
    Square ksq = king_sq[us];

    Bitboard their_diag = bb[them][BISHOP] | bb[them][QUEEN];
    Bitboard their_orth = bb[them][ROOK]   | bb[them][QUEEN];

    for (int i = 0; i < np; i++) {
        Move mv    = pseudo[i];
        Square from = move_from(mv), to = move_to(mv);
        bool is_king   = (piece_on[from] == KING);
        bool is_ep     = (move_flags(mv) == MF_EP);
        bool is_castle = (move_flags(mv) == MF_CASTLE);

        if (!is_king && !is_ep && !is_castle) {
            // Fast legality check: re-run slider attacks on king with new occ
            Bitboard occ_after  = (all & ~sq_bb(from)) | sq_bb(to);
            Bitboard diag_after = their_diag & ~sq_bb(to);
            Bitboard orth_after = their_orth & ~sq_bb(to);
            bool safe =
                !(bishop_attacks(ksq, occ_after) & diag_after) &&
                !(rook_attacks(ksq, occ_after)   & orth_after) &&
                !(g_knight_attacks[ksq] & bb[them][KNIGHT] & ~sq_bb(to)) &&
                !(g_pawn_attacks[us][ksq] & bb[them][PAWN] & ~sq_bb(to)) &&
                !(g_king_attacks[ksq] & bb[them][KING]);
            if (safe) list[legal++] = mv;
        } else {
            // King moves, castling, en-passant: use full do/undo
            Square check_sq = is_king ? to : ksq;
            do_move(mv);
            if (!sq_attacked(check_sq, them)) list[legal++] = mv;
            undo_move(mv);
        }
    }
    return legal;
}

// ─────────────────────────── Draw detection ────────────────────────────────

inline bool Position::is_insufficient_material() const {
    for (int c = 0; c < 2; c++)
        if (bb[c][PAWN] | bb[c][ROOK] | bb[c][QUEEN]) return false;

    int wm = popcount(bb[WHITE][KNIGHT]) + popcount(bb[WHITE][BISHOP]);
    int bm = popcount(bb[BLACK][KNIGHT]) + popcount(bb[BLACK][BISHOP]);
    if (wm == 0 && bm == 0) return true;
    if (wm + bm == 1)       return true;
    // KNN vs K
    if (wm==2 && popcount(bb[WHITE][KNIGHT])==2 && bm==0) return true;
    if (bm==2 && popcount(bb[BLACK][KNIGHT])==2 && wm==0) return true;
    // KBB vs K — draw only if both bishops on same colour
    static constexpr Bitboard LIGHT = 0x55AA55AA55AA55AAULL;
    if (wm==2 && popcount(bb[WHITE][BISHOP])==2 && bm==0) {
        Bitboard wb = bb[WHITE][BISHOP];
        if (!(wb & LIGHT) || !(wb & ~LIGHT)) return true;
    }
    if (bm==2 && popcount(bb[BLACK][BISHOP])==2 && wm==0) {
        Bitboard blb = bb[BLACK][BISHOP];
        if (!(blb & LIGHT) || !(blb & ~LIGHT)) return true;
    }
    return false;
}

inline bool Position::is_draw() const {
    if (cur.halfmove_clock >= 100)    return true;
    if (is_insufficient_material())   return true;
    return is_repetition(history_top);
}

inline bool Position::is_repetition(int root_distance) const {
    int count = 0;
    for (int i = history_top - 2; i >= 0; i -= 2) {
        if (history[i].hash == cur.hash) {
            if (++count >= 2 || i >= root_distance) return true;
        }
        if (history[i].halfmove_clock == 0) break;
    }
    return false;
}

// ─────────────────────────── UCI helpers ───────────────────────────────────

inline Move Position::parse_uci(const std::string &s) const {
    if (s.size() < 4) return NO_MOVE;
    Square from = make_sq(s[0]-'a', s[1]-'1');
    Square to   = make_sq(s[2]-'a', s[3]-'1');
    PieceType promo = NO_PIECE;
    if (s.size() >= 5) {
        switch (s[4]) {
            case 'q': promo=QUEEN;  break; case 'r': promo=ROOK;   break;
            case 'b': promo=BISHOP; break; case 'n': promo=KNIGHT; break;
        }
    }
    MoveFlag flags = MF_NORMAL;
    if (piece_on[from]==PAWN && to==cur.ep_square) flags = MF_EP;
    if (piece_on[from]==KING && std::abs(from-to)==2) flags = MF_CASTLE;
    if (promo != NO_PIECE) flags = MF_PROMO;
    return make_move(from, to, promo, flags);
}

inline std::string Position::move_uci(Move m) const {
    static const char FILES[] = "abcdefgh";
    static const char RANKS[] = "12345678";
    static const char PROMO[] = " pnbrqk";
    Square from = move_from(m), to = move_to(m);
    std::string s;
    s += FILES[sq_file(from)]; s += RANKS[sq_rank(from)];
    s += FILES[sq_file(to)];   s += RANKS[sq_rank(to)];
    if (move_flags(m) == MF_PROMO) s += PROMO[move_promo(m)+1];
    return s;
}
