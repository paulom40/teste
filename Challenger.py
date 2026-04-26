""")

col_text, col_paste = st.columns([2, 1])
with col_text:
    matches_text = st.text_area(
        "Cole aqui os jogos:",
        height=300,
        placeholder="ATP          Lehecka      vs Michelsen    -> Lehecka        61%\nATP          Griekspoor   vs Musetti      -> Musetti        66%\nATP          Prizmic      vs Etcheverry   -> Etcheverry     57%"
    )

with col_paste:
    st.markdown("### Opções")
    surface_override = st.selectbox("Superfície (padrão)", ["Clay", "Hard", "Grass"], index=0)
    if st.button("🔄 LIMPAR", use_container_width=True):
        st.session_state.matches_list = None
        st.rerun()

if matches_text:
    # Parse dos jogos
    parsed_matches = parse_colab_text(matches_text)
    
    if parsed_matches:
        st.success(f"✅ {len(parsed_matches)} jogos detectados!")
        
        # Aplicar superfície padrão se não detectada
        for match in parsed_matches:
            if match['surface'] == 'Clay' and surface_override != 'Clay':
                match['surface'] = surface_override
        
        # Fazer previsões
        if st.button("🎾 FAZER PREVISÕES", type="primary", use_container_width=True):
            results = []
            errors = []
            
            progress_bar = st.progress(0)
            for i, match in enumerate(parsed_matches):
                result, error = predict_match(
                    st.session_state.model, 
                    match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    # Adicionar informação da expectativa da lista
                    if match.get('expected_prob'):
                        result['Prob_Esperada'] = f"{match['expected_prob']:.0%}"
                        result['Favorito_Lista'] = match.get('favorite', '')
                    results.append(result)
                elif error:
                    errors.append(error)
                progress_bar.progress((i + 1) / len(parsed_matches))
            progress_bar.empty()
            
            if errors:
                with st.expander(f"⚠️ {len(errors)} jogadores não encontrados"):
                    for e in set(errors):
                        st.write(e)
            
            if results:
                st.subheader("🎯 RESULTADOS DAS PREVISÕES")
                
                df_results = pd.DataFrame(results)
                
                # Reordenar colunas
                cols = ['Torneio', 'Jogador1', 'Jogador2', 'Match_Historico', 'Superficie',
                       'Prob_P1', 'Prob_P2', 'Vencedor_Previsto', 'Confianca', 'Recomendacao',
                       'Momentum', 'Games_Esperados', 'Dados']
                
                if 'Prob_Esperada' in df_results.columns:
                    cols.insert(6, 'Prob_Esperada')
                    cols.insert(7, 'Favorito_Lista')
                
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                # Aplicar estilo
                def color_recommendation(val):
                    if 'STRONG' in str(val):
                        return 'background-color: #2e7d32; color: white'
                    elif 'GOOD' in str(val):
                        return 'background-color: #4caf50; color: white'
                    elif 'AVOID' in str(val):
                        return 'background-color: #9e9e9e; color: white'
                    return ''
                
                styled = df_results.style.format({
                    'Prob_P1': '{:.1%}',
                    'Prob_P2': '{:.1%}',
                    'Confianca': '{:.1%}'
                }).map(color_recommendation, subset=['Recomendacao'])
                
                st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
                
                # Resumo
                st.subheader("📊 Resumo")
                strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                avoid = sum(1 for r in results if 'AVOID' in r['Recomendacao'])
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("🔥 STRONG", strong)
                with col_s2:
                    st.metric("✅ GOOD", good)
                with col_s3:
                    st.metric("⚪ AVOID", avoid)
                with col_s4:
                    conf_values = [float(r['Confianca'].replace('%', '')) for r in results]
                    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                    st.metric("Confiança Média", f"{avg_conf:.0f}%")
                
                # Comparação com expectativas
                if 'Prob_Esperada' in df_results.columns:
                    st.subheader("📈 Comparação: Lista vs Modelo")
                    comparison = []
                    for r in results:
                        if 'Prob_Esperada' in r:
                            expected = float(r['Prob_Esperada'].replace('%', '')) / 100
                            actual = float(r['Prob_P1'].replace('%', '')) / 100 if 'Prob_P1' in r else 0.5
                            diff = actual - expected
                            comparison.append({
                                'Jogo': f"{r['Jogador1']} vs {r['Jogador2']}",
                                'Prob_Lista': f"{expected:.0%}",
                                'Prob_Modelo': f"{actual:.0%}",
                                'Diferença': f"{diff:+.0%}",
                                'Alinhamento': '✅' if abs(diff) < 0.1 else '⚠️'
                            })
                    df_comp = pd.DataFrame(comparison)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Excel com Previsões",
                    buffer.getvalue(),
                    f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
    else:
        st.warning("⚠️ Nenhum jogo detectado no texto. Use o formato: 'Jogador1 vs Jogador2'")

# Previsão manual individual
with st.expander("✏️ PREVISÃO MANUAL INDIVIDUAL"):
    players_with_stats = [p for p in st.session_state.all_players 
                          if st.session_state.player_stats.get(p, {}).get('matches', 0) > 0]
    players_sorted = sorted(players_with_stats)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        manual_p1 = st.selectbox("Jogador 1", [""] + players_sorted, key="man_p1")
    with col_b:
        manual_p2 = st.selectbox("Jogador 2", [""] + players_sorted, key="man_p2")
    with col_c:
        manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"], key="man_surf")
    
    if st.button("🔮 PREVER JOGO", key="man_btn") and manual_p1 and manual_p2:
        if manual_p1 == manual_p2:
            st.error("Selecione dois jogadores diferentes!")
        else:
            result, error = predict_match(
                st.session_state.model, manual_p1, manual_p2, manual_surface,
                st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                st.session_state.name_matcher
            )
            if result:
                st.dataframe(pd.DataFrame([result]), use_container_width=True)
            else:
                st.error(error)

elif not uploaded_file:
st.info("📂 Faça upload do seu ficheiro Excel/CSV com dados históricos")
st.markdown("""
### Como usar:

1. **Faça upload do seu histórico** (Excel/CSV)
2. **Cole a lista de jogos** no formato:
