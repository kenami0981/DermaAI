using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Derma.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddSkinAnalysisTags : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AnalysisResult",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    skinAnalysisId = table.Column<Guid>(type: "TEXT", nullable: false),
                    acneScore = table.Column<double>(type: "REAL", nullable: false),
                    acneCount = table.Column<int>(type: "INTEGER", nullable: false),
                    outputImageUrl = table.Column<string>(type: "TEXT", nullable: true),
                    tta_applied = table.Column<bool>(type: "INTEGER", nullable: true),
                    sahi_applied = table.Column<bool>(type: "INTEGER", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnalysisResult", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AnalysisResult_SkinAnalysis_skinAnalysisId",
                        column: x => x.skinAnalysisId,
                        principalTable: "SkinAnalysis",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "SkinAnalysisTag",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    SkinAnalysisId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Tag = table.Column<int>(type: "INTEGER", nullable: false),
                    Value = table.Column<bool>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_SkinAnalysisTag", x => x.Id);
                    table.ForeignKey(
                        name: "FK_SkinAnalysisTag_SkinAnalysis_SkinAnalysisId",
                        column: x => x.SkinAnalysisId,
                        principalTable: "SkinAnalysis",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Detection",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AnalysisResultId = table.Column<Guid>(type: "TEXT", nullable: false),
                    className = table.Column<string>(type: "TEXT", nullable: false),
                    bbox = table.Column<string>(type: "TEXT", nullable: false),
                    confidence = table.Column<double>(type: "REAL", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Detection", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Detection_AnalysisResult_AnalysisResultId",
                        column: x => x.AnalysisResultId,
                        principalTable: "AnalysisResult",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AnalysisResult_skinAnalysisId",
                table: "AnalysisResult",
                column: "skinAnalysisId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Detection_AnalysisResultId",
                table: "Detection",
                column: "AnalysisResultId");

            migrationBuilder.CreateIndex(
                name: "IX_SkinAnalysisTag_SkinAnalysisId",
                table: "SkinAnalysisTag",
                column: "SkinAnalysisId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Detection");

            migrationBuilder.DropTable(
                name: "SkinAnalysisTag");

            migrationBuilder.DropTable(
                name: "AnalysisResult");
        }
    }
}
