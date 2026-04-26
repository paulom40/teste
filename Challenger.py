""")

default_surface = st.selectbox("Superficie padrao", ["Clay", "Hard", "Grass"], key="batch_surface")

matches_text = st.text_area(
    "Cole aqui os jogos (um por linha):",
    height=300,
    placeholder="Lehecka vs Michelsen\nGriekspoor vs Musetti\nPrizmic vs Etcheverry"
)

if st.button("PREVER LISTA", type="primary", use_container_width=True):
    if matches_text.strip():
        matches_list = parse_match_text(matches_text, default_surface)
        
        if matches_list:
            st.info(f"{len(matches_list)} jogos para prever")
            
            results = []
            errors = []
            
            progress_bar = st.progress(0)
            for i, match in enumerate(matches_list):
                result, error = predict_match(
                    st.session_state.model, match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                    st.session_state.name_matcher, "Batch"
                )
                if result:
                    results.append(result)
                elif error:
                    errors.append(f"{match['player1']} vs {match['player2']}: {error}")
                progress_bar.progress((i + 1) / len(matches_list))
            progress_bar.empty()
            
            if errors:
                with st.expander(f"{len(errors)} jogadores nao encontrados"):
                    for e in errors[:20]:
                        st.write(e)
            
            if results:
                st.subheader("RESULTADOS")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Summary
                st.subheader("Resumo")
                strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                weak = sum(1 for r in results if 'WEAK' in r['Recomendacao'])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("STRONG", strong)
                c2.metric("GOOD", good)
                c3.metric("WEAK", weak)
                c4.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("Download Excel", buffer.getvalue(),
                                 f"previsoes_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                 use_container_width=True)
        else:
            st.warning("Nenhum jogo encontrado no texto. Use o formato 'Jogador1 vs Jogador2'")
    else:
        st.warning("Cole a lista de jogos no campo acima")

elif not uploaded_file:
st.info("Upload do ficheiro Excel/CSV com dados historicos")
st.markdown("""
### Como preparar o arquivo:

O arquivo deve conter:
- `winner` ou `vencedor` - nome do vencedor
- `loser` ou `perdedor` - nome do perdedor

**Opcional:**
- `date` / `data` - para forma recente
- `score` / `placar` - para total de games
""")

if __name__ == "__main__":
main()
