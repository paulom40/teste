""")
else:
st.success(f"{len(matches)} jogos encontrados!")
results = []
for match in matches:
    pred = predict_match(
        st.session_state.model,
        match['player1'], match['player2'],
        st.session_state.player_stats,
        st.session_state.h2h,
        st.session_state.glicko
    )
    pred['Torneio'] = match['tournament']
    results.append(pred)

if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    buffer = io.BytesIO()
    df_results.to_excel(buffer, index=False)
    st.download_button("📥 Download Previsões", buffer.getvalue(),
                     f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

# Sugestão de jogadores para previsão rápida
with st.expander("💡 Sugestões de jogadores (do seu histórico)"):
players_list = sorted(list(st.session_state.player_stats.keys()))
st.write(f"Total: {len(players_list)} jogadores")
st.write("Exemplos:", ", ".join(players_list[:20]))

if __name__ == "__main__":
main()
