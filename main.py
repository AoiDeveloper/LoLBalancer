import discord
from discord.commands import Option
from discord.ext import commands
import os
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from keep_alive import keep_alive

keep_alive()
# ---------------- 定数・パラメータ設定 ----------------

BOT_TOKEN = os.getenv("DISCORD_TOKEN") 

RANK_TO_MMR = {
    'アイアン4': 0, 'アイアン3': 100, 'アイアン2': 200, 'アイアン1': 300,
    'ブロンズ4': 400, 'ブロンズ3': 500, 'ブロンズ2': 600, 'ブロンズ1': 700,
    'シルバー4': 800, 'シルバー3': 900, 'シルバー2': 1000, 'シルバー1': 1100,
    'ゴールド4': 1200, 'ゴールド3': 1300, 'ゴールド2': 1400, 'ゴールド1': 1500,
    'プラチナ4': 1600, 'プラチナ3': 1700, 'プラチナ2': 1800, 'プラチナ1': 1900,
    'エメラルド4': 2000, 'エメラルド3': 2100, 'エメラルド2': 2200, 'エメラルド1': 2300,
    'ダイヤモンド4': 2400, 'ダイヤモンド3': 2600, 'ダイヤモンド2': 2800, 'ダイヤモンド1': 3000,
    'マスター': 3200, 'グランドマスター': 3500, 'チャレンジャー': 3800
}
ENTRY_LANES = ["TOP", "JG", "MID", "ADC", "SUP"]

INITIAL_TEMP = 1000.0
COOLING_RATE = 0.998
MAX_ITERATIONS = 50000

W_MMR_STD = 1.0
W_LANE_DIV = 100.0
W_TOP_STD = 500.0

# ---------------- データ構造とグローバル変数 ----------------

@dataclass
class Player:
    id: int
    name: str
    rank: str
    role: str
    mmr: int

entry_list: Dict[int, Player] = {}

# ---------------- チーム分けアルゴリズム (変更なし) ----------------
# (省略... 前回のコードと同じものをここに配置してください)
def calculate_multi_team_score(teams: List[List[Player]], all_players: List[Player]) -> float:
    if not teams: return 0.0
    team_mmr_sums = [sum(p.mmr for p in team) for team in teams]
    e_mmr_std = np.std(team_mmr_sums)
    e_lane_diversity = 0
    for team in teams:
        roles_in_team = {p.role for p in team if p.role in ENTRY_LANES}
        e_lane_diversity += (5 - len(roles_in_team))
    num_teams = len(teams)
    top_player_count = min(len(all_players), num_teams * 2)
    top_players = sorted(all_players, key=lambda p: p.mmr, reverse=True)[:top_player_count]
    top_player_distribution = [len(set(p.id for p in team) & set(p.id for p in top_players)) for team in teams]
    e_top_std = np.std(top_player_distribution)
    score = (W_MMR_STD * e_mmr_std) + (W_LANE_DIV * e_lane_diversity) + (W_TOP_STD * e_top_std)
    return score
def balance_multiple_teams(players: List[Player]) -> Tuple[List[List[Player]], List[Player], float]:
    num_teams = len(players) // 5
    num_remains = len(players) % 5
    if num_teams == 0: return [], players, 0.0
    random.shuffle(players)
    team_members = players[:-num_remains] if num_remains > 0 else players
    current_remains = players[-num_remains:] if num_remains > 0 else []
    current_teams = [team_members[i:i+5] for i in range(0, len(team_members), 5)]
    best_teams, best_remains = current_teams, current_remains
    current_score = calculate_multi_team_score(current_teams, players)
    best_score = current_score
    temp = INITIAL_TEMP
    for i in range(MAX_ITERATIONS):
        if temp <= 0.01: break
        new_teams = [list(team) for team in current_teams]
        new_remains = list(current_remains)
        if num_remains > 0 and random.random() < 0.3:
            team_idx, player_idx_in_team, remain_idx = random.randint(0, num_teams - 1), random.randint(0, 4), random.randint(0, num_remains - 1)
            new_teams[team_idx][player_idx_in_team], new_remains[remain_idx] = new_remains[remain_idx], new_teams[team_idx][player_idx_in_team]
        else:
            if num_teams < 2: continue
            t1_idx, t2_idx = random.sample(range(num_teams), 2)
            p1_idx, p2_idx = random.randint(0, 4), random.randint(0, 4)
            new_teams[t1_idx][p1_idx], new_teams[t2_idx][p2_idx] = new_teams[t2_idx][p2_idx], new_teams[t1_idx][p1_idx]
        new_score = calculate_multi_team_score(new_teams, players)
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_teams, current_remains = new_teams, new_remains
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_teams, best_remains = current_teams, current_remains
        temp *= COOLING_RATE
    return best_teams, best_remains, best_score
# -----------------------------------------------------------


# ---------------- Discordボット本体 ----------------

bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user}としてログインしました。")

# --- オートコンプリート用関数 ---
async def rank_autocomplete(ctx: discord.AutocompleteContext):
    all_ranks = list(RANK_TO_MMR.keys())
    user_input = ctx.value.lower()
    return [rank for rank in all_ranks if rank.lower().startswith(user_input)]

# --- コマンド群 ---
@bot.slash_command(name="entry", description="カスタムゲームに参加登録します。")
async def entry(ctx: discord.ApplicationContext, rank: Option(str, "ランクを入力(例:ゴールド)", autocomplete=rank_autocomplete, required=True), role: Option(str, "希望ロールを選択", choices=ENTRY_LANES, required=True)):
    if rank not in RANK_TO_MMR:
        await ctx.respond(f"エラー: '{rank}' は無効なランク名です。候補から選択してください。", ephemeral=True); return
    user = ctx.author
    player = Player(id=user.id, name=user.display_name, rank=rank, role=role, mmr=RANK_TO_MMR[rank])
    is_update = user.id in entry_list
    entry_list[user.id] = player
    action_text = "を更新しました" if is_update else "を受け付けました"
    embed = discord.Embed(title="✅ エントリー完了", description=f"**{user.display_name}** さんのエントリーを{action_text}。", color=discord.Color.blue())
    embed.add_field(name="ランク", value=rank, inline=True); embed.add_field(name="希望ロール", value=role, inline=True)
    embed.set_footer(text=f"現在のエントリー人数: {len(entry_list)}人")
    await ctx.respond(embed=embed, ephemeral=True)

@bot.slash_command(name="withdraw", description="エントリーを取り下げます。")
async def withdraw(ctx: discord.ApplicationContext):
    user_id = ctx.author.id
    if user_id in entry_list:
        del entry_list[user_id]
        await ctx.respond("エントリーを取り下げました。", ephemeral=True)
    else:
        await ctx.respond("あなたはエントリーしていません。", ephemeral=True)

@bot.slash_command(name="status", description="現在のエントリー状況を確認します。")
async def status(ctx: discord.ApplicationContext):
    if not entry_list:
        await ctx.respond("現在、エントリーしている人はいません。", ephemeral=True); return
    embed = discord.Embed(title="エントリー状況", description=f"現在のエントリー人数: **{len(entry_list)}**人", color=discord.Color.gold())
    player_texts = []
    sorted_players = sorted(entry_list.values(), key=lambda p: (ENTRY_LANES.index(p.role), -p.mmr))
    for p in sorted_players: player_texts.append(f"`{p.name:<15}` | {p.rank:<12} | **{p.role}**")
    embed.add_field(name="エントリー一覧", value="\n".join(player_texts) or "なし", inline=False)
    role_counts = {role: 0 for role in ENTRY_LANES}
    for player in entry_list.values(): role_counts[player.role] += 1
    max_count = max(role_counts.values()) if role_counts else 0
    role_status_texts = [f"**{role}**: {count}人 (不足: {max_count - count}人)" for role, count in role_counts.items()]
    embed.add_field(name="ロール状況", value="\n".join(role_status_texts), inline=False)
    await ctx.respond(embed=embed)

class ConfirmView(discord.ui.View):
    def __init__(self, author_id):
        super().__init__(timeout=30); self.value = None; self.author_id = author_id
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("コマンドを実行した本人のみ操作できます。", ephemeral=True); return False
        return True
    @discord.ui.button(label="実行", style=discord.ButtonStyle.green)
    async def confirm(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = True; self.stop()
        for child in self.children: child.disabled = True
        await interaction.response.edit_message(content="処理を実行します...", view=self)
    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.grey)
    async def cancel(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = False; self.stop()
        for child in self.children: child.disabled = True
        await interaction.response.edit_message(content="キャンセルしました。", view=self)

def create_result_embed(player_list: List[Player], teams: List[List[Player]], remains: List[Player], score: float) -> discord.Embed:
    embed = discord.Embed(title="チーム分け結果", description=f"参加者 {len(player_list)}名から {len(teams)}チームを編成しました。", color=discord.Color.green())
    def format_team(team: List[Player]):
        avg_mmr = sum(p.mmr for p in team) / len(team) if team else 0
        text = f"**平均MMR: {avg_mmr:.0f}**\n"
        sorted_team = sorted(team, key=lambda p: ENTRY_LANES.index(p.role) if p.role in ENTRY_LANES else 99)
        for p in sorted_team: text += f"- `{p.name}` ({p.rank} / {p.role})\n"
        return text
    for i, team in enumerate(teams):
        embed.add_field(name=f"【チーム {chr(65+i)}】", value=format_team(team), inline=True)
    if remains:
        remain_text = "\n".join([f"- `{p.name}` ({p.rank} / {p.role})" for p in remains])
        embed.add_field(name="待機メンバー", value=remain_text, inline=False)
    embed.set_footer(text=f"最適化スコア: {score:.2f} (低いほど良い)")
    return embed

@bot.slash_command(name="divide_teams", description="現在エントリー中のメンバーでチーム分けを行います。")
@commands.has_permissions(administrator=True)
async def divide_teams(ctx: discord.ApplicationContext):
    if len(entry_list) < 5:
        await ctx.respond(f"エラー: 5人未満ではチーム分けできません。(現在: {len(entry_list)}人)", ephemeral=True); return
    view = ConfirmView(ctx.author.id)
    await ctx.respond(f"現在のエントリー {len(entry_list)} 人でチーム分けを実行します。\n**実行後、エントリーリストはクリアされます。**", view=view, ephemeral=True)
    await view.wait()
    if view.value is True:
        player_list = list(entry_list.values())
        entry_list.clear()
        teams, remains, score = balance_multiple_teams(player_list)
        result_embed = create_result_embed(player_list, teams, remains, score)
        await ctx.followup.send(embed=result_embed)
        await ctx.followup.send("エントリーリストをクリアしました。次の募集を開始できます。", ephemeral=True)

@bot.slash_command(name="clear_entries", description="エントリーリストを強制的にリセットします。")
@commands.has_permissions(administrator=True)
async def clear_entries(ctx: discord.ApplicationContext):
    entry_list.clear()
    await ctx.respond("✅ エントリーリストをリセットしました。", ephemeral=True)

## NEW ##
@bot.slash_command(name="help", description="BOTのコマンド一覧と使い方を表示します。")
async def help(ctx: discord.ApplicationContext):
    embed = discord.Embed(title=" LoLカスタムチーム分けBOT ヘルプ", description="このBOTで利用できるコマンドの一覧です。", color=discord.Color.og_blurple())
    
    embed.add_field(
        name="【👤 参加者向けコマンド】",
        value=(
            f"`/entry` `rank:<ランク>` `role:<ロール>`\n"
            "カスタムに参加登録します。ランク入力時には候補が表示されます。\n\n"
            f"`/withdraw`\n"
            "カスタムへの参加登録を取り下げます。\n\n"
            f"`/status`\n"
            "現在のエントリー状況（参加者一覧、ロール状況）を確認します。\n\n"
            f"`/help`\n"
            "このヘルプメッセージを表示します。"
        ),
        inline=False
    )
    
    embed.add_field(
        name="【👑 管理者向けコマンド】",
        value=(
            f"`/divide_teams`\n"
            "現在エントリーしているメンバーで、チーム分けを実行します。\n\n"
            f"`/clear_entries`\n"
            "すべてのエントリー情報をリセットします。"
        ),
        inline=False
    )
    
    embed.add_field(
        name="【🧪 テスト用コマンド】",
        value=(
            f"`/debug` `count:<人数>`\n"
            "テスト用のダミー参加者データを生成します。"
        ),
        inline=False
    )
    
    embed.set_footer(text="困ったときはこのコマンドを実行してください。")
    await ctx.respond(embed=embed)


# ---------------- デバッグ用コマンド (変更なし) ----------------
# (省略... 前回のdebugコマンドのコードをここにコピーしてください)


# ボットの実行
bot.run(BOT_TOKEN)
