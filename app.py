import io
import itertools

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle


# ---------------------------------------------------------------------------
# CIELab–sRGB Gamut Plotter — Streamlit web application
# Original project:
# https://github.com/UlrichKlotz/CIELab_Plotter
# DOI: https://doi.org/10.5281/zenodo.22811675
# ---------------------------------------------------------------------------

APP_DOI = "10.5281/zenodo.22811675"
APP_GITHUB = "https://github.com/UlrichKlotz/CIELab_Plotter"

# --- Autor / Zitierung -----------------------------------------------------
AUTHOR = "Prof. Dr. Ulrich E. Klotz"
AFFILIATION = "Hochschule München University of Applied Sciences"
APP_YEAR = 2026
APP_TITLE_FULL = "CIELab–sRGB Gamut Plotter"
# Empfohlene Software-Zitierung (bitte Jahr und DOI vor Veröffentlichung prüfen).
CITATION = (
    f"{AUTHOR} ({APP_YEAR}). {APP_TITLE_FULL} [Computer software]. "
    f"{AFFILIATION}. https://doi.org/{APP_DOI}"
)

MARKER_POOL = ["o", "s", "^", "D", "v", "p", "h", "<", ">", "*"]
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#ffe119", "#000000", "#e6beff",
]

_M_XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_D65 = np.array([0.95047, 1.00000, 1.08883])


def _f_inv(t):
    delta = 6 / 29
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))


def lab_to_xyz(L, a, b):
    L = np.asarray(L, float) * np.ones_like(a)
    fy = (L + 16) / 116
    fx = np.asarray(a, float) / 500 + fy
    fz = fy - np.asarray(b, float) / 200
    return (
        _D65[0] * _f_inv(fx),
        _D65[1] * _f_inv(fy),
        _D65[2] * _f_inv(fz),
    )


def xyz_to_srgb(X, Y, Z):
    rgb = np.einsum(
        "ij,...j->...i",
        _M_XYZ2RGB,
        np.stack([X, Y, Z], axis=-1),
    )
    c = np.clip(rgb, 0, None)
    return np.clip(
        np.where(
            c <= 0.0031308,
            12.92 * c,
            1.055 * c ** (1 / 2.4) - 0.055,
        ),
        0,
        1,
    )


@st.cache_data(show_spinner=False)
def build_background(L, a0, a1, b0, b1, resolution=500, oog_grey=0.93):
    """Create the true sRGB gamut background at a fixed L*."""
    a = np.linspace(a0, a1, resolution)
    b = np.linspace(b0, b1, resolution)
    aa, bb = np.meshgrid(a, b)

    X, Y, Z = lab_to_xyz(L, aa, bb)
    lin_rgb = np.einsum(
        "ij,...j->...i",
        _M_XYZ2RGB,
        np.stack([X, Y, Z], axis=-1),
    )
    in_gamut = np.all(
        (lin_rgb >= -0.001) & (lin_rgb <= 1.001),
        axis=-1,
    )

    img = xyz_to_srgb(X, Y, Z)
    img[~in_gamut] = oog_grey
    return img


def parse_data(text):
    """Parse the same basic tab/semicolon/comma format as the desktop app."""
    rows = []
    skipped = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "\t" in line:
            sep = "\t"
        elif ";" in line:
            sep = ";"
        else:
            sep = ","

        parts = [c.strip() for c in line.split(sep)]

        if len(parts) < 5:
            skipped += 1
            continue

        head = parts[0].lower()
        if head.startswith("composition") or head in ("label", "comp"):
            continue

        try:
            comp = parts[0] or "Sample"
            phase = parts[1]
            Lp = float(parts[2])
            a = float(parts[3])
            b = float(parts[4])
        except (ValueError, TypeError):
            skipped += 1
            continue

        rows.append(
            {
                "Composition": comp,
                "Phase": phase,
                "L*": Lp,
                "a*": a,
                "b*": b,
                "Show": True,
                "Label": False,
            }
        )

    if not rows:
        raise ValueError(
            "Keine gültigen Datenzeilen gefunden. "
            "Erwartet werden mindestens fünf Spalten: "
            "Composition, Phase, L*, a*, b*."
        )

    return pd.DataFrame(rows), skipped


def demo_dataframe():
    return pd.DataFrame(
        [
            ["Au 99.99", "Au", 86.00, 4.70, 36.90, True, False],
            ["Ag 99.99", "Ag", 92.65, -0.31, 5.05, True, False],
            ["Cu 99.99", "Cu", 87.20, 13.40, 14.90, True, False],
        ],
        columns=["Composition", "Phase", "L*", "a*", "b*", "Show", "Label"],
    )


def normalise_dataframe(df):
    """Clean data after st.data_editor and validate numeric columns."""
    df = df.copy()

    required = ["Composition", "Phase", "L*", "a*", "b*"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt.")

    for col in ["L*", "a*", "b*"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Composition"] = df["Composition"].fillna("").astype(str)
    df["Phase"] = df["Phase"].fillna("").astype(str)

    if "Show" not in df.columns:
        df["Show"] = True
    if "Label" not in df.columns:
        df["Label"] = False

    df["Show"] = df["Show"].fillna(True).astype(bool)
    df["Label"] = df["Label"].fillna(False).astype(bool)

    bad = df[["L*", "a*", "b*"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} Datenzeile(n) enthalten ungültige Zahlen "
            "in L*, a* oder b*."
        )

    df = df[df["Composition"].str.strip() != ""].reset_index(drop=True)
    return df


def make_styles(df):
    styles = {}
    colour_cycle = itertools.cycle(PALETTE)
    marker_cycle = itertools.cycle(MARKER_POOL)

    for comp in df["Composition"].drop_duplicates():
        styles[comp] = {
            "colour": next(colour_cycle),
            "marker": next(marker_cycle),
        }
    return styles


def add_phase_label(ax, row, label_size):
    if not row["Label"] or not row["Phase"]:
        return

    ax.annotate(
        row["Phase"],
        (row["a*"], row["b*"]),
        textcoords="offset points",
        xytext=(7, 5),
        fontsize=label_size,
        color="#111",
        ha="left",
        va="bottom",
        zorder=7,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def _add_source_note(fig, ax, legend, export):
    """Quellen-/DOI-Hinweis direkt unterhalb der Legende platzieren.

    Der Text wird auf zwei Zeilen gesetzt, linksbündig mit der Legende
    ausgerichtet und – falls nötig – so weit verkleinert, dass er nicht
    breiter als die Legende ist (mit einer lesbaren Mindestgröße).
    """
    note = f"Generated with CIELab Plotter\nDOI: {APP_DOI}"

    # Legende muss gezeichnet sein, bevor ihre Position gemessen werden kann.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    lbox = legend.get_window_extent(renderer)
    lx0, ly0 = inv.transform((lbox.x0, lbox.y0))   # untere linke Ecke
    lx1, _ = inv.transform((lbox.x1, lbox.y1))     # rechte Kante
    legend_width = lx1 - lx0

    gap = 0.015                       # vertikaler Abstand unter der Legende
    fontsize = 7.5 if export else 7.0

    txt = ax.text(
        lx0,
        ly0 - gap,
        note,
        transform=ax.transAxes,
        fontsize=fontsize,
        color="#444",
        ha="left",
        va="top",
        linespacing=1.35,
        clip_on=False,
        zorder=6,
    )

    # Schrift verkleinern, bis der Hinweis in die Legendenbreite passt.
    for _ in range(10):
        fig.canvas.draw()
        tbox = txt.get_window_extent(renderer)
        tx0, _ = inv.transform((tbox.x0, tbox.y0))
        tx1, _ = inv.transform((tbox.x1, tbox.y1))
        if (tx1 - tx0) <= legend_width or fontsize <= 5.0:
            break
        fontsize -= 0.5
        txt.set_fontsize(fontsize)


def create_figure(df, L_background, a_min, a_max, b_min, b_max,
                  label_size=9, export=False):
    """Create the publication-style Matplotlib figure."""
    figsize = (8.0, 8.0) if export else (7.2, 7.2)
    fig = plt.figure(figsize=figsize, facecolor="white")

    # Space on the right is reserved for the legend.
    fig.subplots_adjust(
        left=0.12,
        right=0.74,
        top=0.91,
        bottom=0.14 if export else 0.12,
    )
    ax = fig.add_subplot(111)

    resolution = 600 if export else 450
    img = build_background(
        int(L_background),
        float(a_min),
        float(a_max),
        float(b_min),
        float(b_max),
        resolution=resolution,
    )

    ax.imshow(
        img,
        extent=[a_min, a_max, b_min, b_max],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )

    if a_min <= 0 <= a_max:
        ax.axvline(0, color="#333", lw=0.7, ls="--", alpha=0.6)
    if b_min <= 0 <= b_max:
        ax.axhline(0, color="#333", lw=0.7, ls="--", alpha=0.6)

    styles = make_styles(df)

    # Plot visible points.
    for _, row in df.iterrows():
        if not row["Show"]:
            continue

        style = styles[row["Composition"]]
        marker = style["marker"]
        colour = style["colour"]

        if MarkerStyle(marker).is_filled():
            ax.scatter(
                row["a*"],
                row["b*"],
                c=colour,
                marker=marker,
                s=120 if export else 100,
                edgecolors="#222",
                linewidths=0.8,
                zorder=5,
            )
        else:
            ax.scatter(
                row["a*"],
                row["b*"],
                c=colour,
                marker=marker,
                s=120 if export else 100,
                linewidths=1.3,
                zorder=5,
            )

        add_phase_label(ax, row, label_size + (1 if export else 0))

    # One legend entry per composition.
    handles = []
    seen = set()
    for _, row in df.iterrows():
        if not row["Show"]:
            continue
        comp = row["Composition"]
        if comp in seen:
            continue
        seen.add(comp)

        style = styles[comp]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["colour"],
                markeredgecolor="#222",
                markersize=9,
                label=comp,
            )
        )

    if handles:
        ncol = 1 if len(handles) <= 16 else 2
        legend = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            ncol=ncol,
            fontsize=8 if export else 8,
            framealpha=1.0,
            edgecolor="#ccc",
        )

    ax.set_box_aspect(1)
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(b_min, b_max)
    ax.set_xlabel("a*   (green − / + red)", fontsize=13 if export else 11)
    ax.set_ylabel("b*   (blue − / + yellow)", fontsize=13 if export else 11)
    ax.set_title(
        f"CIELab a*–b*  at  L* = {int(L_background)}",
        fontsize=14 if export else 12,
        pad=10,
    )
    ax.tick_params(labelsize=11 if export else 9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Quellen-/DOI-Hinweis zweizeilig direkt unter der Legende. Bewusst erst
    # jetzt, nachdem set_box_aspect() die endgültige Achsengeometrie festgelegt
    # hat, damit die gemessene Legendenposition korrekt ist.
    # Teil der Matplotlib-Figur und damit auch im exportierten PNG.
    if handles:
        _add_source_note(fig, ax, legend, export)

    return fig


@st.cache_data(show_spinner=False, max_entries=32)
def render_png(df, L_background, a_min, a_max, b_min, b_max,
               label_size, export, dpi):
    """Diagramm als PNG-Bytes erzeugen und zwischenspeichern.

    Solange sich Daten und Einstellungen nicht ändern, liefert Streamlit
    bei jedem erneuten Durchlauf das gecachte Ergebnis zurück, statt die
    (teure) Matplotlib-Figur neu zu berechnen. Das verhindert das Ruckeln.
    """
    fig = create_figure(
        df,
        L_background,
        a_min,
        a_max,
        b_min,
        b_max,
        label_size=label_size,
        export=export,
    )
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CIELab–sRGB Gamut Plotter",
    page_icon="🎨",
    layout="wide",
)

st.title("CIELab–sRGB Gamut Plotter")
st.caption(
    "Visualisierung von CIELab-Messdaten im a*–b*-Diagramm "
    "mit dem tatsächlichen sRGB-Gamut."
)

st.markdown(f"**Autor:** {AUTHOR} · {AFFILIATION}")
st.caption("Empfohlene Zitierung dieser Software:")
st.code(CITATION, language="text")

with st.sidebar:
    st.header("Darstellung")

    L_background = st.slider(
        "L* Hintergrund",
        min_value=1,
        max_value=99,
        value=80,
        help="Helligkeit L* der dargestellten sRGB-Gamut-Ebene.",
    )

    st.subheader("a* Bereich")
    a_min, a_max = st.slider(
        "a* min / max",
        min_value=-128,
        max_value=128,
        value=(-60, 60),
        step=1,
    )

    st.subheader("b* Bereich")
    b_min, b_max = st.slider(
        "b* min / max",
        min_value=-128,
        max_value=128,
        value=(-60, 60),
        step=1,
    )

    label_size = st.slider(
        "Phasenbezeichnung",
        min_value=5,
        max_value=24,
        value=9,
    )

    st.divider()
    st.markdown(
        f"**Quelle des Programms**  \n"
        f"[CIELab Plotter auf GitHub]({APP_GITHUB})  \n"
        f"DOI: [10.5281/zenodo.22811675]"
        f"(https://doi.org/{APP_DOI})"
    )

# Session state: no save/load session feature is exposed.
if "data" not in st.session_state:
    st.session_state.data = demo_dataframe()

# Input section
st.subheader("1. Messdaten eingeben")

left, right = st.columns([1.4, 1])

with left:
    pasted = st.text_area(
        "Daten aus Excel / LibreOffice / Textdatei einfügen",
        height=180,
        placeholder=(
            "Composition\\tPhase\\tL*\\ta*\\tb*\\n"
            "Au 99.99\\tAu\\t86\\t4.7\\t36.9\\n"
            "Ag 99.99\\tAg\\t92.65\\t-0.31\\t5.05"
        ),
        help=(
            "Erlaubt sind Tabulator, Semikolon oder Komma als Trennzeichen. "
            "Die Kopfzeile ist optional."
        ),
    )

    import_col1, import_col2 = st.columns(2)

    with import_col1:
        if st.button("Daten importieren", type="primary", use_container_width=True):
            if not pasted.strip():
                st.warning("Bitte zuerst Daten einfügen.")
            else:
                try:
                    new_df, skipped = parse_data(pasted)
                    st.session_state.data = new_df
                    msg = f"{len(new_df)} Datenpunkt(e) importiert."
                    if skipped:
                        msg += f" {skipped} Zeile(n) wurden übersprungen."
                    st.success(msg)
                except ValueError as exc:
                    st.error(str(exc))

    with import_col2:
        if st.button("Beispieldaten laden", use_container_width=True):
            st.session_state.data = demo_dataframe()
            st.success("Beispieldaten geladen.")

with right:
    uploaded = st.file_uploader(
        "Oder Datei hochladen",
        type=["csv", "tsv", "txt"],
        help="CSV-, TSV- oder TXT-Datei mit mindestens fünf Spalten.",
    )

    if uploaded is not None:
        # Avoid importing the same uploaded file on every rerun.
        file_signature = (uploaded.name, uploaded.size)
        if st.session_state.get("uploaded_signature") != file_signature:
            try:
                text = uploaded.getvalue().decode("utf-8-sig")
                new_df, skipped = parse_data(text)
                st.session_state.data = new_df
                st.session_state.uploaded_signature = file_signature
                msg = f"{len(new_df)} Datenpunkt(e) importiert."
                if skipped:
                    msg += f" {skipped} Zeile(n) wurden übersprungen."
                st.success(msg)
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Datei konnte nicht gelesen werden: {exc}")

# Data editor
st.subheader("2. Daten kontrollieren und bearbeiten")

edited = st.data_editor(
    st.session_state.data,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "Composition": st.column_config.TextColumn(
            "Composition",
            help="Eindeutige Bezeichnung der Zusammensetzung.",
        ),
        "Phase": st.column_config.TextColumn("Phase"),
        "L*": st.column_config.NumberColumn(
            "L*",
            format="%.2f",
        ),
        "a*": st.column_config.NumberColumn(
            "a*",
            format="%.2f",
        ),
        "b*": st.column_config.NumberColumn(
            "b*",
            format="%.2f",
        ),
        "Show": st.column_config.CheckboxColumn(
            "Anzeigen",
            help="Datenpunkt im Diagramm anzeigen.",
        ),
        "Label": st.column_config.CheckboxColumn(
            "Phase beschriften",
            help="Phasenbezeichnung neben dem Datenpunkt anzeigen.",
        ),
    },
    key="data_editor",
)

try:
    edited = normalise_dataframe(edited)
    st.session_state.data = edited
except ValueError as exc:
    st.error(str(exc))
    edited = None

# Plot
if edited is not None and not edited.empty:
    st.subheader("3. Diagramm")

    # Anzeige-Diagramm (gecacht -> kein Neuberechnen bei jeder Interaktion).
    display_png = render_png(
        edited,
        L_background,
        a_min,
        a_max,
        b_min,
        b_max,
        label_size=label_size,
        export=False,
        dpi=130,
    )
    st.image(display_png)

    # Export-Diagramm in Publikationsqualität (ebenfalls gecacht).
    export_png = render_png(
        edited,
        L_background,
        a_min,
        a_max,
        b_min,
        b_max,
        label_size=label_size,
        export=True,
        dpi=200,
    )

    st.download_button(
        label="PNG herunterladen",
        data=export_png,
        file_name=f"cielab_L{int(L_background)}.png",
        mime="image/png",
        type="primary",
    )

else:
    st.info("Bitte mindestens einen gültigen Datenpunkt eingeben.")
