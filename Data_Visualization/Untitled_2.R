# ============================================================
#  Eurovision Rankings Visualisation
#  Charts:
#    1. poll_Vs_televote.png  – Spearman similarity over years
#    2. public_Change.png     – Poll → Televote bump chart
#    3. Jury_effect.png       – Televote → Jury → Final bump chart
# ============================================================

# ---- Libraries ----
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(ggrepel)
library(scales)
library(patchwork)
library(ggtext)

# ---- Parameters ----
csv_file   <- "eurovision_results3.csv"
save_path1 <- "poll_Vs_televote.png"   # Chart 1: similarity over time
save_path2 <- "public_Change.png"      # Chart 2: poll → televote bump
save_path3 <- "Jury_effect.png"        # Chart 3: televote → jury → final bump

dpi    <- 300
width  <- 12
height <- 10

col_year     <- "Year"
col_country  <- "Country"
col_poll     <- "Poll_Rank"
col_televote <- "Place_Televote"
col_jury     <- "Place_Jury"
col_final    <- "Place"

# ---- Shared theme & palette ----
theme_euro <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor    = element_blank(),
    panel.grid.major.x  = element_blank(),
    axis.text           = element_text(colour = "grey30"),
    plot.title          = element_text(size = 22, face = "bold", colour = "#1a1a2e"),
    plot.subtitle       = element_markdown(size = 13, colour = "grey40", lineheight = 1.4),
    plot.caption        = element_text(size = 9, colour = "grey60", hjust = 0),
    legend.position     = "right",
    legend.title        = element_text(size = 11, face = "bold"),
    legend.text         = element_text(size = 10),
    plot.margin         = margin(16, 16, 16, 16)
  )

clr_improve  <- "#1a9850"   # green  = improved rank (lower number)
clr_worsen   <- "#d73027"   # red    = worsened rank (higher number)
clr_neutral  <- "grey80"

# ---- Load & clean data ----
data_all <- read_csv(csv_file, show_col_types = FALSE)

# Handle alternate Televote column name
if (!col_televote %in% names(data_all) && "Televote_Rank" %in% names(data_all)) {
  data_all   <- data_all %>% rename(Place_Televote = Televote_Rank)
  col_televote <- "Place_Televote"
}

data_filtered <- data_all %>%
  filter(
    .data[[col_year]] >= 2016,
    .data[[col_year]] <= 2023,
    .data[[col_year]] != 2020
  ) %>%
  mutate(
    Poll_Rank     = as.numeric(.data[[col_poll]]),
    Televote_Rank = as.numeric(.data[[col_televote]]),
    Jury_Rank     = as.numeric(.data[[col_jury]]),
    Final_Rank    = as.numeric(.data[[col_final]])
  ) %>%
  select(
    Year      = .data[[col_year]],
    Country   = .data[[col_country]],
    Poll_Rank, Televote_Rank, Jury_Rank, Final_Rank
  )

# ==============================================================
# CHART 1 – Poll vs Televote Similarity by Year
#   Story: "Were Eurovision fans accurate predictors of the
#           public vote? And in which year did they diverge most?"
# ==============================================================

spearman_by_year <- data_filtered %>%
  filter(!is.na(Televote_Rank), !is.na(Poll_Rank)) %>%
  group_by(Year) %>%
  summarise(
    spearman_corr = if (n() >= 2)
      cor(Televote_Rank, Poll_Rank, method = "spearman")
    else NA_real_,
    .groups = "drop"
  ) %>%
  arrange(Year)

# Year the public surprised pollsters most
year_low  <- spearman_by_year %>% filter(!is.na(spearman_corr)) %>%
  slice_min(spearman_corr, n = 1) %>% pull(Year)
year_high <- spearman_by_year %>% filter(!is.na(spearman_corr)) %>%
  slice_max(spearman_corr, n = 1) %>% pull(Year)

p_spearman <- ggplot(spearman_by_year, aes(x = Year, y = spearman_corr)) +
  # Shaded reference band for "strong agreement"
  annotate("rect",
           xmin = -Inf, xmax = Inf, ymin = 0.7, ymax = 1,
           fill = clr_improve, alpha = 0.07) +
  annotate("text",
           x = min(spearman_by_year$Year, na.rm = TRUE),
           y = 0.72,
           label = "Strong agreement zone",
           hjust = 0, colour = clr_improve, size = 3.5, fontface = "italic") +
  # Line + area fill
  geom_area(fill = "#4575b4", alpha = 0.12) +
  geom_line(colour = "#4575b4", linewidth = 1.6) +
  # Points coloured by value
  geom_point(aes(colour = spearman_corr), size = 6, stroke = 0) +
  # Annotate worst & best years
  geom_label(
    data = spearman_by_year %>% filter(Year == year_low),
    aes(label = paste0(Year, "\nLowest match\n", round(spearman_corr, 2))),
    vjust = -0.4, size = 3.8, colour = clr_worsen,
    fill = "white", label.size = 0.3, label.padding = unit(0.3, "lines")
  ) +
  geom_label(
    data = spearman_by_year %>% filter(Year == year_high),
    aes(label = paste0(Year, "\nHighest match\n", round(spearman_corr, 2))),
    vjust = 1.5, size = 3.8, colour = clr_improve,
    fill = "white", label.size = 0.3, label.padding = unit(0.3, "lines")
  ) +
  scale_colour_gradient(
    low = clr_worsen, high = clr_improve,
    limits = c(0, 1),
    breaks = c(0, 0.5, 1),
    labels = c("No match", "Moderate", "Perfect"),
    name   = "Agreement\n(Spearman ρ)"
  ) +
  scale_x_continuous(breaks = c(2016:2019, 2021:2022)) +
  scale_y_continuous(
    limits = c(0, 1.05),
    labels = label_number(accuracy = 0.1)
  ) +
  labs(
    title    = "Did Fan Polls Predict the Public Vote?",
    subtitle = paste0(
      "Spearman rank correlation between pre-contest poll rankings and final televote results (2016–2022).\n",
      "<span style='color:", clr_improve, ";'><b>Green</b></span> = fans & viewers agreed; ",
      "<span style='color:", clr_worsen, ";'><b>Red</b></span> = the public surprised the pollsters."
    ),
    x       = NULL,
    y       = "Spearman ρ (0 = no match, 1 = perfect match)",
    caption = "Excludes 2020 (no contest). Each point = one Eurovision final."
  ) +
  theme_euro

ggsave(save_path1, p_spearman, width = width, height = height * 0.75, dpi = dpi)
message("Saved: ", save_path1)

# ==============================================================
# CHART 2 – Public Sentiment Shift: Poll → Televote (focus year)
#   Story: "In the year polls were most wrong, which countries
#           the public unexpectedly loved – or ignored?"
# ==============================================================

year_focus <- year_low   # use the year of lowest poll-televote agreement

data_focus <- data_filtered %>%
  filter(Year == year_focus) %>%
  mutate(
    televote_effect = Televote_Rank - Poll_Rank,   # positive = worsened
    jury_effect     = Final_Rank    - Televote_Rank,
    abs_televote    = abs(televote_effect),
    abs_jury        = abs(jury_effect)
  )

# Threshold: flag countries with above-median shift as "notable"
threshold_tv <- median(data_focus$abs_televote, na.rm = TRUE)

data_focus <- data_focus %>%
  mutate(
    tv_category = case_when(
      televote_effect < -threshold_tv ~ "improved",
      televote_effect >  threshold_tv ~ "worsened",
      TRUE                            ~ "neutral"
    )
  )

# Long format for bump chart
data_poll_tv <- data_focus %>%
  select(Country, Poll_Rank, Televote_Rank, televote_effect, abs_televote, tv_category) %>%
  pivot_longer(
    cols      = c(Poll_Rank, Televote_Rank),
    names_to  = "Stage",
    values_to = "Rank"
  ) %>%
  mutate(
    Stage = factor(Stage,
                   levels = c("Poll_Rank", "Televote_Rank"),
                   labels = c("Fan Poll", "Televote"))
  )

# For labelling: one row per country per side
labels_left  <- data_poll_tv %>% filter(Stage == "Fan Poll")
labels_right <- data_poll_tv %>% filter(Stage == "Televote")

max_rank_tv <- max(data_poll_tv$Rank, na.rm = TRUE)
max_abs_tv  <- max(data_focus$abs_televote, na.rm = TRUE)

p_poll_tv <- ggplot(data_poll_tv,
                    aes(x = Stage, y = Rank, group = Country)) +
  geom_line(
    aes(colour    = televote_effect,
        linewidth = abs_televote,
        alpha     = tv_category)
  ) +
  # Country labels – left side
  geom_text(
    data  = labels_left,
    aes(label = Country, colour = televote_effect),
    hjust = 1.08, size = 3.5
  ) +
  # Country labels – right side
  geom_text(
    data  = labels_right,
    aes(label = Country, colour = televote_effect),
    hjust = -0.08, size = 3.5
  ) +
  scale_colour_gradient2(
    low      = clr_improve,
    mid      = clr_neutral,
    high     = clr_worsen,
    midpoint = 0,
    limits   = c(-max_abs_tv, max_abs_tv),
    breaks   = c(-max_abs_tv, 0, max_abs_tv),
    labels   = c("Improved", "No change", "Worsened"),
    name     = "Rank change\n(Poll → Televote)"
  ) +
  scale_alpha_manual(
    values = c("improved" = 1, "worsened" = 1, "neutral" = 0.22),
    guide  = "none"
  ) +
  scale_linewidth_continuous(
    range = c(0.4, 2.2),
    guide = "none"
  ) +
  scale_x_discrete(expand = expansion(add = c(1.5, 1.5))) +
  scale_y_reverse(limits = c(max_rank_tv + 0.5, 0.5)) +
  coord_cartesian(clip = "off") +
  labs(
    title    = paste0("When Viewers Disagreed with the Polls (", year_focus, ")"),
    subtitle = paste0(
      "Bump chart: each line is one country. Rank 1 = winner (top).\n",
      "<span style='color:", clr_improve, ";'><b>Green lines</b></span> = the public ranked them <b>higher</b> than polls predicted; ",
      "<span style='color:", clr_worsen,  ";'><b>red lines</b></span> = lower. ",
      "Faded lines = little change. Line thickness = size of shift."
    ),
    x       = NULL,
    y       = NULL,
    caption = paste0("Focus year: ", year_focus,
                     " — the year with the lowest poll–televote agreement (ρ = ",
                     round(spearman_by_year$spearman_corr[spearman_by_year$Year == year_focus], 2), ").")
  ) +
  theme_euro +
  theme(
    axis.text.y      = element_blank(),
    panel.grid.major = element_blank(),
    plot.margin      = margin(16, 80, 16, 80)
  )

ggsave(save_path2, p_poll_tv, width = width, height = height, dpi = dpi)
message("Saved: ", save_path2)

# ==============================================================
# CHART 3 – Jury Effect: Televote → Jury → Final (focus year)
#   Story: "The jury is supposed to balance popular taste – but
#           did it help or hurt the public's favourites?"
# ==============================================================

threshold_jury <- median(data_focus$abs_jury, na.rm = TRUE)

data_focus <- data_focus %>%
  mutate(
    jury_category = case_when(
      jury_effect < -threshold_jury ~ "improved",
      jury_effect >  threshold_jury ~ "worsened",
      TRUE                          ~ "neutral"
    )
  )

data_tv_jury <- data_focus %>%
  select(Country, Televote_Rank, Jury_Rank, Final_Rank,
         jury_effect, abs_jury, jury_category) %>%
  pivot_longer(
    cols      = c(Televote_Rank, Jury_Rank, Final_Rank),
    names_to  = "Stage",
    values_to = "Rank"
  ) %>%
  mutate(
    Stage = factor(Stage,
                   levels = c("Televote_Rank", "Jury_Rank", "Final_Rank"),
                   labels = c("Televote", "Jury", "Final"))
  )

labels_tv_left  <- data_tv_jury %>% filter(Stage == "Televote")
labels_tv_right <- data_tv_jury %>% filter(Stage == "Final")
points_jury     <- data_tv_jury %>% filter(Stage == "Jury")

max_rank_jury <- max(data_tv_jury$Rank, na.rm = TRUE)
max_abs_jury  <- max(data_focus$abs_jury, na.rm = TRUE)

p_tv_jury <- ggplot(data_tv_jury,
                    aes(x = Stage, y = Rank, group = Country)) +
  geom_line(
    aes(colour    = jury_effect,
        linewidth = abs_jury,
        alpha     = jury_category)
  ) +
  # Jury position dot
  geom_point(
    data  = points_jury,
    aes(colour = jury_effect),
    size  = 3,
    shape = 21,
    fill  = "white",
    stroke = 1.2,
    show.legend = FALSE
  ) +
  # Labels
  geom_text(
    data  = labels_tv_left,
    aes(label = Country, colour = jury_effect),
    hjust = 1.08, size = 3.5
  ) +
  geom_text(
    data  = labels_tv_right,
    aes(label = Country, colour = jury_effect),
    hjust = -0.08, size = 3.5
  ) +
  scale_colour_gradient2(
    low      = clr_improve,
    mid      = clr_neutral,
    high     = clr_worsen,
    midpoint = 0,
    limits   = c(-max_abs_jury, max_abs_jury),
    breaks   = c(-max_abs_jury, 0, max_abs_jury),
    labels   = c("Jury helped", "No net effect", "Jury hurt"),
    name     = "Jury's net\nrank change"
  ) +
  scale_alpha_manual(
    values = c("improved" = 1, "worsened" = 1, "neutral" = 0.22),
    guide  = "none"
  ) +
  scale_linewidth_continuous(
    range = c(0.4, 2.2),
    guide = "none"
  ) +
  scale_x_discrete(expand = expansion(add = c(1.5, 1.5))) +
  scale_y_reverse(limits = c(max_rank_jury + 0.5, 0.5)) +
  coord_cartesian(clip = "off") +
  labs(
    title    = paste0("Did the Jury Reflect the People? (", year_focus, ")"),
    subtitle = paste0(
      "Three stages: what the public voted (Televote), what the jury scored (Jury), and the combined result (Final).\n",
      "<span style='color:", clr_improve, ";'><b>Green</b></span> = jury <b>boosted</b> the country's final rank; ",
      "<span style='color:", clr_worsen,  ";'><b>red</b></span> = jury <b>dragged it down</b>. ",
      "The open dot marks each jury rank."
    ),
    x       = NULL,
    y       = NULL,
    caption = "Rank 1 = winner (top of chart). Line thickness reflects magnitude of jury impact."
  ) +
  theme_euro +
  theme(
    axis.text.y      = element_blank(),
    panel.grid.major = element_blank(),
    plot.margin      = margin(16, 80, 16, 80)
  )

ggsave(save_path3, p_tv_jury, width = width, height = height, dpi = dpi)
message("Saved: ", save_path3)

message("\nAll three charts saved successfully.")